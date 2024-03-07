import numpy as np
from numpy import array

class DataSet:
    """
    The class containing the data for the neural network training, and test
    """
    def __init__(self, 
                dataset_name: str = "",
                description: str = "",
                parameters: dict = {},
                x_train: np.array = [],
                x_test: np.array = [], 
                y_train: np.array = [],
                y_test: np.array = []):
        """
        
        """
        self.name: str = dataset_name
        self.description: str = description
        self.parameters: dict = parameters
        self.x_train: np.array  = x_train
        self.x_test: np.array  = x_test
        self.y_train: np.array  = y_train
        self.y_test: np.array  = y_test
        
        self.x_train_shape = self.x_train.shape
        self.y_train_shape = self.y_train.shape
        self.x_test_shape = self.x_test.shape
        self.y_test_shape = self.y_test.shape
        
    def get_metadata(self):
        return {"name":self.name, "description": self.description, "parameters": self.parameters}
        
    def normalize(self):
        """
        Must be implemented
        """
        pass
    
    
class ImageToImageDataSet(DataSet):
    
    def __init__(self, dataset_name: str = "", description: str = "", parameters: dict = {},
                 x_train: np.array = [], x_test: np.array = [], y_train: np.array = [], y_test: np.array = [],
                 max_val = None, min_val = None):
        super().__init__(dataset_name, description, parameters, x_train, x_test, y_train, y_test)
        
        self.max_val = max_val
        self.min_val = min_val
    
    def normalize (self):
        """
        Normalize a image dividing by its maximum value
        """
        
        if not self.max_val:
            self.max_val = np.max(self.x_train)
            
        if not self.min_val:
            self.min_val = np.min(self.x_train)
            
        self.x_train = self.x_train/self.max_val
        self.y_train = self.y_train/self.max_val
        self.x_test = self.x_test/self.max_val
        self.y_test = self.y_test/self.max_val
        
        self.name += "_Norm"
            
        
        
class InpaintingDataSet(ImageToImageDataSet):
    
    def __init__(self, mask = None, dataset_name: str = "", description: str = "", parameters: dict = {},
                 x_train: np.array = [], x_test: np.array = [], y_train: np.array = [], y_test: np.array = [],
                 max_val=None, min_val=None):
        super().__init__(dataset_name, description, parameters, x_train, x_test, y_train, y_test, max_val, min_val)

        if mask:
            self.apply_mask_to_data(mask)
    
    def apply_mask_to_data(self, mask_function):
        
        self.x_train = mask_function(self.x_train_shape)*self.x_train
        self.x_test = mask_function(self.x_test_shape)*self.x_test
        
        

class LocalInpaintingDataSet (InpaintingDataSet):
    
    def __init__(self, mask=None, dataset_name: str = "rafael_cifar_10_64x64_compressed", description: str = "An upsampled cirfar 10 compressed with loss of quality",
                 parameters: dict = {}, max_val=None, min_val=None):
        
        x_train = np.load("/home/rafaeltadeu/old/autoencoder/X_64x64_treino.npy").astype('float32')
        x_test = np.load("/home/rafaeltadeu/old/autoencoder/X_64x64_teste.npy").astype('float32')
        y_train = np.load("/home/rafaeltadeu/old/autoencoder/Y_64x64_treino.npy").astype('float32')
        y_test = np.load("/home/rafaeltadeu/old/autoencoder/Y_64x64_teste.npy").astype('float32')
        
        
        super().__init__(mask, dataset_name, description, parameters, x_train, x_test, y_train, y_test, max_val, min_val)