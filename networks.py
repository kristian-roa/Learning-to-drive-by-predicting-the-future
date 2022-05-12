from __future__ import print_function
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, Add, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# tf.python.control_flow_ops = tf

class Networks(object):

    @staticmethod
    def dfp_network(input_shape, measurement_size, goal_size, action_size, num_timesteps, learning_rate, lr_step, lr_decay):
        """
        Neural Network for Direct Future Predition (DFP)
        """

        # Perception Feature
        state_input = Input(shape=input_shape)
        perception_feat = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(state_input)
        perception_feat = Convolution2D(64, (4, 4), strides=(2, 2), activation='relu')(perception_feat)
        perception_feat = Convolution2D(64, (3, 3), activation='relu')(perception_feat)
        perception_feat = Flatten()(perception_feat)
        perception_feat = Dense(512, activation='relu')(perception_feat)

        # Measurement Feature
        measurement_input = Input(shape=(measurement_size,))
        measurement_feat = Dense(128, activation='relu')(measurement_input)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)
        measurement_feat = Dense(128, activation='relu')(measurement_feat)

        # Goal Feature
        goal_input = Input(shape=(goal_size,))
        goal_feat = Dense(128, activation='relu')(goal_input)
        goal_feat = Dense(128, activation='relu')(goal_feat)
        goal_feat = Dense(128, activation='relu')(goal_feat)

        concat_feat = Concatenate()([perception_feat, measurement_feat, goal_feat])

        measurement_pred_size = measurement_size * num_timesteps  # 4 measurements, 6 timesteps

        expectation_stream = Dense(measurement_pred_size, activation='relu')(concat_feat)

        prediction_list = []
        for i in range(action_size):
            action_stream = Dense(measurement_pred_size, activation='relu')(concat_feat)
            prediction_list.append(Add()([action_stream, expectation_stream]))

        model = Model(inputs=[state_input, measurement_input, goal_input], outputs=prediction_list)

        lr = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=lr_step, decay_rate=lr_decay, staircase=True)
        adam = Adam(learning_rate=lr, beta_1=0.95, beta_2=0.999, epsilon=0.0001)
        model.compile(loss='mse', optimizer=adam)

        return model
