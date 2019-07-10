import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D, AveragePooling2D, concatenate
from keras.layers import Conv2D, Dense, Flatten, SeparableConv2D


class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0
        self.model = None

    def build_model(self, input_image, center_map, batch_size):

        inputs = Input(shape=(368, 368, 3))

        net = inputs
        net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                     activation='relu', name='sub_stages/sub_conv1')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                     activation='relu', name='sub_stages/sub_conv2')(net)
        net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool1')(net)

        net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                     activation='relu', name='sub_stages/sub_conv3')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                     activation='relu', name='sub_stages/sub_conv4')(net)
        net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool2')(net)

        net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                     activation='relu', name='sub_stages/sub_conv5')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                     activation='relu', name='sub_stages/sub_conv6')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                     activation='relu', name='sub_stages/sub_conv7')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
                     activation='relu', name='sub_stages/sub_conv8')(net)
        net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool3')(net)

        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv9')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv10')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv11')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv12')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv13')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
                     activation='relu', name='sub_stages/sub_conv14')(net)
        net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
                     activation='relu', name='sub_stages/sub_stage_img_feature')(net)

        self.sub_stage_img_feature = net

        net = Conv2D(kernel_size=1, strides=1, filters=512, padding='same',
                     activation='relu', name='stage_1/conv1')(net)
        net = Conv2D(kernel_size=1, strides=1, filters=self.joints, padding='same',
                     activation='relu', name='stage_1/stage_heatmap')(net)

        self.stage_heatmap.append(net)

        # outputs = None

        for stage in range(2, self.stages + 1):
            self.current_featuremap = concatenate([self.stage_heatmap[stage - 2],
                                                   self.sub_stage_img_feature,
                                                   # self.center_map,
                                                   ],
                                                  axis=3)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv1')(self.current_featuremap)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv2')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv3')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv4')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv5')(net)
            net = Conv2D(kernel_size=1, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv6')(net)
            self.current_heatmap = Conv2D(kernel_size=1, strides=1, filters=self.joints, padding='same',
                                          activation='relu', name='stage_' + str(stage) + '/mid_conv7')(net)
            self.stage_heatmap.append(self.current_heatmap)
            # outputs = self.current_heatmap

        print("------------------final current heatmap------------------------------")
        print(self.current_heatmap)

        print("------------------final stage heatmap------------------------------")
        print(self.stage_heatmap)

        model = Model(inputs=inputs, outputs=[self.stage_heatmap[0],
                                              self.stage_heatmap[1],
                                              self.stage_heatmap[2],
                                              self.stage_heatmap[3],
                                              self.stage_heatmap[4],
                                              self.stage_heatmap[5]])

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def build_model_mobilenet(self):
        inputs = Input(shape=(368, 368, 3))

        net = inputs
        net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                     activation='relu', name='sub_stages/sub_conv1')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
        #              activation='relu', name='sub_stages/sub_conv2')(net)
        # net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool1')(net)
        # net = SeparableConv2D(kernel_size=3, strides=2, filters=64, padding='same', activation='relu6',
        #              name='sub_stages/sub_dw_conv2')(net)


        # net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
        #              activation='relu', name='sub_stages/sub_conv3')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu6',
                     name='sub_stages/sub_dw_conv3')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
        #              activation='relu', name='sub_stages/sub_conv4')(net)
        # net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool2')(net)
        net = SeparableConv2D(kernel_size=3, strides=2, filters=128, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv4')(net)


        # net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
        #              activation='relu', name='sub_stages/sub_conv5')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv5')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
        #              activation='relu', name='sub_stages/sub_conv6')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv6')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
        #              activation='relu', name='sub_stages/sub_conv7')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=256, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv7')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=256, padding='same',
        #              activation='relu', name='sub_stages/sub_conv8')(net)
        # net = MaxPooling2D(pool_size=2, name='sub_stages/sub_pool3')(net)
        net = SeparableConv2D(kernel_size=3, strides=2, filters=256, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv8')(net)

        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv9')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv9')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv10')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv10')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv11')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv11')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv12')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv12')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv13')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv13')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=512, padding='same',
        #              activation='relu', name='sub_stages/sub_conv14')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=512, padding='same', activation='relu6',
                              name='sub_stages/sub_dw_conv14')(net)
        # net = Conv2D(kernel_size=3, strides=1, filters=128, padding='same',
        #              activation='relu', name='sub_stages/sub_stage_img_feature')(net)
        net = SeparableConv2D(kernel_size=3, strides=1, filters=128, padding='same', activation='relu6',
                              name='sub_stages/ssub_stage_img_feature_dw')(net)

        self.sub_stage_img_feature = net

        net = Conv2D(kernel_size=1, strides=1, filters=512, padding='same',
                     activation='relu', name='stage_1/conv1')(net)
        net = Conv2D(kernel_size=1, strides=1, filters=self.joints, padding='same',
                     activation='relu', name='stage_1/stage_heatmap')(net)

        self.stage_heatmap.append(net)

        # outputs = None

        for stage in range(2, self.stages + 1):
            self.current_featuremap = concatenate([self.stage_heatmap[stage - 2],
                                                   self.sub_stage_img_feature,
                                                   # self.center_map,
                                                   ],
                                                  axis=3)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv1')(self.current_featuremap)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv2')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv3')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv4')(net)
            net = Conv2D(kernel_size=7, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv5')(net)
            net = Conv2D(kernel_size=1, strides=1, filters=128, padding='same',
                         activation='relu', name='stage_' + str(stage) + '/mid_conv6')(net)
            self.current_heatmap = Conv2D(kernel_size=1, strides=1, filters=self.joints, padding='same',
                                          activation='relu', name='stage_' + str(stage) + '/mid_conv7')(net)
            self.stage_heatmap.append(self.current_heatmap)
            # outputs = self.current_heatmap

        print("------------------final current heatmap------------------------------")
        print(self.current_heatmap)

        print("------------------final stage heatmap------------------------------")
        print(self.stage_heatmap)

        model = Model(inputs=inputs, outputs=[self.stage_heatmap[0],
                                              self.stage_heatmap[1],
                                              self.stage_heatmap[2],
                                              self.stage_heatmap[3],
                                              self.stage_heatmap[4],
                                              self.stage_heatmap[5]])

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    # def build_loss(self, gt_heatmap, lr, lr_decay_rate, lr_decay_step):
    #     self.gt_heatmap = gt_heatmap
    #     self.total_loss = 0
    #     self.learning_rate = lr
    #     self.lr_decay_rate = lr_decay_rate
    #     self.lr_decay_step = lr_decay_step
    #
    #     for stage in range(self.stages):
    #         print("=====stage heatmap shape")
    #         print(tf.shape(self.stage_heatmap[stage]))
    #
    #         with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
    #             self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
    #                                                    name='l2_loss') / self.batch_size
    #         tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])
    #
    #     with tf.variable_scope('total_loss'):
    #         for stage in range(self.stages):
    #             self.total_loss += self.stage_loss[stage]
    #         tf.summary.scalar('total loss', self.total_loss)
    #
    #     with tf.variable_scope('train'):
    #         self.global_step = tf.contrib.framework.get_or_create_global_step()
    #
    #         self.lr = tf.train.exponential_decay(self.learning_rate,
    #                                              global_step=self.global_step,
    #                                              decay_rate=self.lr_decay_rate,
    #                                              decay_steps=self.lr_decay_step)
    #         tf.summary.scalar('learning rate', self.lr)
    #
    #         self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
    #                                                         global_step=self.global_step,
    #                                                         learning_rate=self.lr,
    #                                                         optimizer='Adam')
    #     self.merged_summary = tf.summary.merge_all()

    def set_weights(self, model_vars):
        model = self.model

        for i in range(1, 15):
            layer = model.get_layer(name='sub_stages/sub_conv' + str(i))
            layer.set_weights([model_vars['sub_stages/sub_conv' + str(i) + '/weights:0'],
                               model_vars['sub_stages/sub_conv' + str(i) + '/biases:0']])

        layer = model.get_layer(name='sub_stages/sub_stage_img_feature')
        layer.set_weights([model_vars['sub_stages/sub_stage_img_feature/weights:0'],
                           model_vars['sub_stages/sub_stage_img_feature/biases:0']])

        layer = model.get_layer(name='stage_1/conv1')
        layer.set_weights([model_vars['stage_1/conv1/weights:0'],
                           model_vars['stage_1/conv1/biases:0']])

        layer = model.get_layer(name='stage_1/stage_heatmap')
        layer.set_weights([model_vars['stage_1/stage_heatmap/weights:0'],
                           model_vars['stage_1/stage_heatmap/biases:0']])

        for i in range(2, 7):
            for j in range(1, 8):
                layer = model.get_layer(name='stage_' + str(i) + '/mid_conv' + str(j))
                layer.set_weights([model_vars['stage_' + str(i) + '/mid_conv' + str(j) + '/weights:0'],
                                   model_vars['stage_' + str(i) + '/mid_conv' + str(j) + '/biases:0']])

    def predict(self, input_image):
        model = self.model

        self.stage_heatmap[0], self.stage_heatmap[1], \
        self.stage_heatmap[2], self.stage_heatmap[3], self.stage_heatmap[4], \
        self.stage_heatmap[5] = model.predict(x=input_image)
        print("--------------------------prediction result by keras---------------------------")
        print(self.current_heatmap)
        print("--------------------------prediction result shape---------------------------")
        print(self.current_heatmap.shape)
        print("--------------------------end of output info---------------------------")
        # self.stage_heatmap.append(self.current_heatmap)
        # self.stage_heatmap.append(self.current_heatmap)
        # self.stage_heatmap.append(self.current_heatmap)
        # self.stage_heatmap.append(self.current_heatmap)
        # self.stage_heatmap.append(self.current_heatmap)
        # self.stage_heatmap.append(self.current_heatmap)

        return self.stage_heatmap[5], self.stage_heatmap

    def save(self, path):
        model = self.model
        model.save(path)
        # -----------------------------print weights
        for weight in model.weights:
            print(weight)
