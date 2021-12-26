# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data preprocessing and augmentation."""

# code picked up from https://github.com/google-research/simclr

import functools

import tensorflow as tf


CROP_PROPORTION = 1

class preprocess_image(object):

    def __init__(self, img_size, color_distort = True, test_crop = True, is_training = True):
        self.height = img_size
        self.width = img_size
        self.color_distort = color_distort
        self.test_crop = test_crop
        self.is_training = is_training
        self.seed = (1, 11)
        self.clip_min = 0.
        self.clip_max = 255.

        if img_size <= 32:
            self.test_crop = False


    def random_apply(self, func, p, x):
        """Randomly apply function func to x with probability p."""
        return tf.cond(
            tf.less(
                tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)), lambda: func(x), lambda: x)


    def random_brightness(self, image, max_delta, impl='v2'):
        """A multiplicative vs additive change of brightness."""
        if impl == 'v2':
            factor = tf.random.uniform([], tf.maximum(1.0 - max_delta, 0),
                                       1.0 + max_delta)
            image = image * factor
        elif impl == 'v1':
            image = tf.image.stateless_random_brightness(image, max_delta=max_delta, seed = self.seed)
        else:
            raise ValueError('Unknown impl {} for random brightness.'.format(impl))
        return image


    def to_grayscale(self, image, keep_channels=True):
        image = tf.image.stateless_random_saturation(image, 0.0, 0.05, seed = self.seed)
        # if keep_channels:
        #     image = tf.tile(image, [1, 1, 3])
        return image


    def color_jitter(self, image, strength, random_order=True, impl='v2'):
        """Distorts the color of the image.

        Args:
          image: The input image tensor.
          strength: the floating number for the strength of the color augmentation.
          random_order: A bool, specifying whether to randomize the jittering order.
          impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
              version of random brightness.

        Returns:
          The distorted image tensor.
        """
        brightness = 0.8 * strength
        contrast = 0.8 * strength
        saturation = 0.8 * strength
        hue = 0.2 * strength
        if random_order:
            return self.color_jitter_rand(
                image, brightness, contrast, saturation, hue, impl=impl)
        else:
            return self.color_jitter_nonrand(
                image, brightness, contrast, saturation, hue, impl=impl)


    def color_jitter_nonrand(self, image,
                             brightness=0,
                             contrast=0,
                             saturation=0,
                             hue=0,
                             impl='v2'):
        """Distorts the color of the image (jittering order is fixed).

        Args:
          image: The input image tensor.
          brightness: A float, specifying the brightness for color jitter.
          contrast: A float, specifying the contrast for color jitter.
          saturation: A float, specifying the saturation for color jitter.
          hue: A float, specifying the hue for color jitter.
          impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
              version of random brightness.

        Returns:
          The distorted image tensor.
        """
        with tf.name_scope('distort_color'):
            def apply_transform(i, x, brightness, contrast, saturation, hue):
                """Apply the i-th transformation."""
                if brightness != 0 and i == 0:
                    x = self.random_brightness(x, max_delta=brightness, impl=impl)
                elif contrast != 0 and i == 1:
                    x = tf.image.stateless_random_contrast(
                        x, lower=1-contrast, upper=1+contrast, seed = self.seed)
                elif saturation != 0 and i == 2:
                    x = tf.image.stateless_random_saturation(
                        x, lower=1-saturation, upper=1+saturation, seed = self.seed)
                elif hue != 0:
                    x = tf.image.stateless_random_hue(x, max_delta=hue, seed = self.seed)
                return x

            for i in range(4):
                image = apply_transform(i, image, brightness, contrast, saturation, hue)
                image = tf.clip_by_value(image, self.clip_min, self.clip_max)
            return image


    def color_jitter_rand(self, image,
                          brightness=0,
                          contrast=0,
                          saturation=0,
                          hue=0,
                          impl='v2'):
        """Distorts the color of the image (jittering order is random).

        Args:
          image: The input image tensor.
          brightness: A float, specifying the brightness for color jitter.
          contrast: A float, specifying the contrast for color jitter.
          saturation: A float, specifying the saturation for color jitter.
          hue: A float, specifying the hue for color jitter.
          impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
              version of random brightness.

        Returns:
          The distorted image tensor.
        """
        with tf.name_scope('distort_color'):
            def apply_transform(i, x):
                """Apply the i-th transformation."""
                def brightness_foo():
                    if brightness == 0:
                        return x
                    else:
                        return self.random_brightness(x, max_delta=brightness, impl=impl)

                def contrast_foo():
                    if contrast == 0:
                        return x
                    else:
                        return tf.image.stateless_random_contrast(x, lower=1-contrast, upper=1+contrast, seed = self.seed)
                def saturation_foo():
                    if saturation == 0:
                        return x
                    else:
                        return tf.image.stateless_random_saturation(x, lower=1-saturation, upper=1+saturation, seed = self.seed)
                def hue_foo():
                    if hue == 0:
                        return x
                    else:
                        return tf.image.stateless_random_hue(x, max_delta=hue, seed = self.seed)
                x = tf.cond(tf.less(i, 2),
                            lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                            lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
                return x

            perm = tf.random.shuffle(tf.range(4))
            for i in range(4):
                image = apply_transform(perm[i], image)
                image = tf.clip_by_value(image, self.clip_min, self.clip_max)
            return image


    def _compute_crop_shape(
            self, image_height, image_width, aspect_ratio, crop_proportion):
        """Compute aspect ratio-preserving shape for central crop.

        The resulting shape retains `crop_proportion` along one side and a proportion
        less than or equal to `crop_proportion` along the other side.

        Args:
          image_height: Height of image to be cropped.
          image_width: Width of image to be cropped.
          aspect_ratio: Desired aspect ratio (width / height) of output.
          crop_proportion: Proportion of image to retain along the less-cropped side.

        Returns:
          crop_height: Height of image after cropping.
          crop_width: Width of image after cropping.
        """
        image_width_float = tf.cast(image_width, tf.float32)
        image_height_float = tf.cast(image_height, tf.float32)

        def _requested_aspect_ratio_wider_than_image():
            crop_height = tf.cast(
                tf.math.rint(crop_proportion / aspect_ratio * image_width_float),
                tf.int32)
            crop_width = tf.cast(
                tf.math.rint(crop_proportion * image_width_float), tf.int32)
            return crop_height, crop_width

        def _image_wider_than_requested_aspect_ratio():
            crop_height = tf.cast(
                tf.math.rint(crop_proportion * image_height_float), tf.int32)
            crop_width = tf.cast(
                tf.math.rint(crop_proportion * aspect_ratio * image_height_float),
                tf.int32)
            return crop_height, crop_width

        return tf.cond(
            aspect_ratio > image_width_float / image_height_float,
            _requested_aspect_ratio_wider_than_image,
            _image_wider_than_requested_aspect_ratio)


    def center_crop(self, image, height, width, crop_proportion):
        """Crops to center of image and rescales to desired size.

        Args:
          image: Image Tensor to crop.
          height: Height of image to be cropped.
          width: Width of image to be cropped.
          crop_proportion: Proportion of image to retain along the less-cropped side.

        Returns:
          A `height` x `width` x channels Tensor holding a central crop of `image`.
        """
        shape = tf.shape(image)
        image_height = shape[0]
        image_width = shape[1]
        crop_height, crop_width = self._compute_crop_shape(
            image_height, image_width, width / height, crop_proportion)
        offset_height = ((image_height - crop_height) + 1) // 2
        offset_width = ((image_width - crop_width) + 1) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, crop_height, crop_width)

        image = tf.image.resize(image, [height, width],
                                method=tf.image.ResizeMethod.BICUBIC)

        return image


    def distorted_bounding_box_crop(self, image,
                                    bbox,
                                    min_object_covered=0.1,
                                    aspect_ratio_range=(0.75, 1.33),
                                    area_range=(0.05, 1.0),
                                    max_attempts=100,
                                    scope=None):
        """Generates cropped_image using one of the bboxes randomly distorted.

        See `tf.image.sample_distorted_bounding_box` for more documentation.

        Args:
          image: `Tensor` of image data.
          bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
              where each coordinate is [0, 1) and the coordinates are arranged
              as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
              image.
          min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
              area of the image must contain at least this fraction of any bounding
              box supplied.
          aspect_ratio_range: An optional list of `float`s. The cropped area of the
              image must have an aspect ratio = width / height within this range.
          area_range: An optional list of `float`s. The cropped area of the image
              must contain a fraction of the supplied image within in this range.
          max_attempts: An optional `int`. Number of attempts at generating a cropped
              region of the image of the specified constraints. After `max_attempts`
              failures, return the entire image.
          scope: Optional `str` for name scope.
        Returns:
          (cropped image `Tensor`, distorted bbox `Tensor`).
        """
        with tf.name_scope(scope or 'distorted_bounding_box_crop'):
            shape = tf.shape(image)
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, _ = sample_distorted_bounding_box

            # Crop the image to the specified bounding box.
            offset_y, offset_x, _ = tf.unstack(bbox_begin)
            target_height, target_width, _ = tf.unstack(bbox_size)
            image = tf.image.crop_to_bounding_box(
                image, offset_y, offset_x, target_height, target_width)

            return image


    def crop_and_resize(self, image, height, width):
        """Make a random crop and resize it to height `height` and width `width`.

        Args:
          image: Tensor representing the image.
          height: Desired image height.
          width: Desired image width.

        Returns:
          A `height` x `width` x channels Tensor holding a random crop of `image`.
        """
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
        aspect_ratio = width / height
        image = self.distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.08, 1.0),
            max_attempts=100,
            scope=None)
        return tf.image.resize(image, [height, width],
                               method=tf.image.ResizeMethod.BICUBIC)


    def gaussian_blur(self, image, kernel_size, sigma, padding='SAME'):
        """Blurs the given image with separable convolution.


        Args:
          image: Tensor of shape [height, width, channels] and dtype float to blur.
          kernel_size: Integer Tensor for the size of the blur kernel. This is should
            be an odd number. If it is an even number, the actual kernel size will be
            size + 1.
          sigma: Sigma value for gaussian operator.
          padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

        Returns:
          A Tensor representing the blurred image.
        """
        radius = tf.cast(kernel_size / 2, dtype=tf.int32)
        kernel_size = radius * 2 + 1
        x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
        blur_filter = tf.exp(-tf.pow(x, 2.0) /
                             (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
        blur_filter /= tf.reduce_sum(blur_filter)
        # One vertical and one horizontal filter.
        blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
        blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
        num_channels = tf.shape(image)[-1]
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        expand_batch_dim = image.shape.ndims == 3
        if expand_batch_dim:
            # Tensorflow requires batched input to convolutions, which we can fake with
            # an extra dimension.
            image = tf.expand_dims(image, axis=0)
        blurred = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding=padding)
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
        if expand_batch_dim:
            blurred = tf.squeeze(blurred, axis=0)
        return blurred


    def random_crop_with_resize(self, image, height, width, p=1.0):
        """Randomly crop and resize an image.

        Args:
          image: `Tensor` representing an image of arbitrary size.
          height: Height of output image.
          width: Width of output image.
          p: Probability of applying this transformation.

        Returns:
          A preprocessed image `Tensor`.
        """
        def _transform(image):  # pylint: disable=missing-docstring
            image = self.crop_and_resize(image, height, width)
            return image
        return self.random_apply(_transform, p=p, x=image)


    def random_color_jitter(self, image, p=1.0, strength=1.0,
                            impl='v2'):

        def _transform(image):
            color_jitter_t = functools.partial(
                self.color_jitter, strength=strength, impl=impl)
            image = self.random_apply(color_jitter_t, p=0.8, x=image)
            return self.random_apply(self.to_grayscale, p=0.2, x=image)
        return self.random_apply(_transform, p=p, x=image)


    def random_blur(self, image, height, width, p=1.0):
        """Randomly blur an image.

        Args:
          image: `Tensor` representing an image of arbitrary size.
          height: Height of output image.
          width: Width of output image.
          p: probability of applying this transformation.

        Returns:
          A preprocessed image `Tensor`.
        """
        del width
        def _transform(image):
            sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
            return self.gaussian_blur(
                image, kernel_size=height//10, sigma=sigma, padding='SAME')
        return self.random_apply(_transform, p=p, x=image)


    def batch_random_blur(self, images_list, height, width, blur_probability=0.5):
        """Apply efficient batch data transformations.

        Args:
          images_list: a list of image tensors.
          height: the height of image.
          width: the width of image.
          blur_probability: the probaility to apply the blur operator.

        Returns:
          Preprocessed feature list.
        """
        def generate_selector(p, bsz):
            shape = [bsz, 1, 1, 1]
            selector = tf.cast(
                tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
                tf.float32)
            return selector

        new_images_list = []
        for images in images_list:
            images_new = self.random_blur(images, height, width, p=1.)
            selector = generate_selector(blur_probability, tf.shape(images)[0])
            images = images_new * selector + images * (1 - selector)
            images = tf.clip_by_value(images, self.clip_min, self.clip_max)
            new_images_list.append(images)

        return new_images_list


    def preprocess_for_train(self, image,
                             height,
                             width,
                             color_distort=True,
                             crop=True,
                             flip=True,
                             impl='v2'):
        """Preprocesses the given image for training.

        Args:
          image: `Tensor` representing an image of arbitrary size.
          height: Height of output image.
          width: Width of output image.
          color_distort: Whether to apply the color distortion.
          crop: Whether to crop the image.
          flip: Whether or not to flip left and right of an image.
          impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
              version of random brightness.

        Returns:
          A preprocessed image `Tensor`.
        """
        if not crop:
            image = self.random_crop_with_resize(image, height, width)
        if not flip:
            image = tf.image.random_flip_left_right(image)
        if color_distort:
            image = self.random_color_jitter(image, strength=0.5,
                                        impl=impl)
        image = tf.image.resize(image, [height, width],
                                method=tf.image.ResizeMethod.BICUBIC)
        image = tf.clip_by_value(image, self.clip_min, self.clip_max)
        return image


    def preprocess_for_eval(self, image, height, width, crop=True):
        """Preprocesses the given image for evaluation.

        Args:
          image: `Tensor` representing an image of arbitrary size.
          height: Height of output image.
          width: Width of output image.
          crop: Whether or not to (center) crop the test images.

        Returns:
          A preprocessed image `Tensor`.
        """
        if crop:
            image = self.center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
        else:
            image = tf.image.resize(image, [height, width],
                                    method=tf.image.ResizeMethod.BICUBIC)
        # Akash commented below line
        #image = tf.reshape(image, [height, width, 3])
        image = tf.clip_by_value(image, self.clip_min, self.clip_max)
        return image


    def __call__(self, image):
        """Preprocesses the given image.

        Args:
          image: `Tensor` representing an image of arbitrary size.
          height: Height of output image.
          width: Width of output image.
          is_training: `bool` for whether the preprocessing is for training.
          color_distort: whether to apply the color distortion.
          test_crop: whether or not to extract a central crop of the images
              (as for standard ImageNet evaluation) during the evaluation.

        Returns:
          A preprocessed image `Tensor` of range [0, 1].
        """
        self.seed = tf.random.experimental.stateless_split(self.seed, num=1)[0, :]
        # Below line is problematic for some reason.
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.cast(image, tf.float32)
        #image = image/255.
        if self.is_training:
            return self.preprocess_for_train(image, self.height, self.width, self.color_distort)
        else:
            return self.preprocess_for_eval(image, self.height, self.width, self.test_crop)
