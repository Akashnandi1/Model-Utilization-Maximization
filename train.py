import tensorflow as tf

def train(cfg, model, train_ds, test_ds):

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01,
                                        momentum=0.9,
                                        nesterov=True,
                                        name='SGD')
    #model = multi_gpu_model(model, gpus=4)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    #model = multi_gpu_model(model, gpus=4)

    # def scheduler(epoch):
    #     if epoch < 31:
    #         return 0.01
    #     elif epoch < 41:
    #         return 0.001
    #     elif epoch < 46:
    #         return 0.0001

    def cust_scheduler(epoch):
        lr = cfg['model']['optimizer']['args']['lr']
        gamma = cfg['model']['scheduler']['args']['gamma']
        milestones = cfg['model']['scheduler']['args']['milestones']
        new_lr = lr/gamma
        for i in milestones:
            new_lr = new_lr*gamma
            if epoch<i:
                return new_lr

    callback = tf.keras.callbacks.LearningRateScheduler(cust_scheduler, verbose = 1)
    callback_list = [callback]

    # Create data generators
    history = model.fit(train_ds, validation_data=test_ds, epochs=cfg['model']['epochs'], callbacks = callback_list)
