from keras.layers import Add, Input, BatchNormalization, Activation, Dense, LeakyReLU
from keras.models import Model

class myResNet:
    def __init__(self, input_shape, output_shape, hidden_node, hidden_layer=18,
                    leakyReLU=False,leaky_alpha=0.01):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_node = hidden_node
        self.hidden_layer = hidden_layer
        if(leakyReLU is False): 
            self.model = self.make_model()
        else: 
            self.leaky_alpha = leaky_alpha
            self.model = self.make_model_leakyReLU()

    def shortcut(self,conv,residual):
        return Add()([conv, residual])

    def res_blocks(self,x): 
        conv = Dense(self.hidden_node)(x)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        conv = Dense(self.hidden_node)(conv)
        conv = BatchNormalization()(conv)
        short_cut = self.shortcut(conv, x)
        conv = Activation("relu")(short_cut)
        return conv

    def make_model(self):
        if(self.hidden_layer % 2 != 0): print("self.hidden_layer must be odd.")
        inputs = Input(self.input_shape)

        x = Dense(self.hidden_node)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        for _ in range(int((self.hidden_layer -2)/2)):
            # print(x,self.hidden_node)
            x = self.res_blocks(x)
        
        outputs = Dense(units=self.output_shape, activation='linear')(x)
        ResNetModel = Model(inputs=inputs, outputs=outputs)
        return ResNetModel


    def res_blocks_leakyReLU(self,x): 
        conv = Dense(self.hidden_node)(x)
        conv = BatchNormalization()(conv)
        conv = LeakyReLU(alpha=self.leaky_alpha)(conv)

        conv = Dense(self.hidden_node)(conv)
        conv = BatchNormalization()(conv)
        short_cut = self.shortcut(conv, x)
        conv = LeakyReLU(alpha=self.leaky_alpha)(short_cut)
        return conv
    
    def make_model_leakyReLU(self):
        if(self.hidden_layer % 2 != 0): print("self.hidden_layer must be odd.")
        inputs = Input(self.input_shape)
        x = Dense(self.hidden_node)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leaky_alpha)(x)

        for _ in range(int((self.hidden_layer -2)/2)):
            x = self.res_blocks_leakyReLU(x)
        
        outputs = Dense(units=self.output_shape, activation='linear')(x)
        ResNetModel = Model(inputs=inputs, outputs=outputs)
        return ResNetModel


def build(input_shape, output_shape, hidden_node, hidden_layer=18, leakyReLU=False,leaky_alpha=0.01):
    return myResNet(input_shape, output_shape, hidden_node, hidden_layer=hidden_layer,leakyReLU=leakyReLU,leaky_alpha=leaky_alpha).model

# model = ResNet18(input_shape=(224,224,3), nb_classes=1000).model