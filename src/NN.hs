module NN where
import System.Random (randomRIO)
import Control.Monad (replicateM)
import Data.Matrix
import System.IO.Unsafe (unsafePerformIO)
import System.IO
import Control.Monad (foldM)  -- Assuming both are used

-- Function to generate a matrix with random values between a and b, appearing as pure
generateWeightsMatrix :: (Int, Int) -> Double -> Double -> Matrix Double
generateWeightsMatrix (n, m) a b = unsafePerformIO $ fromLists <$> replicateM n (replicateM m (randomRIO (a, b)))

-- Function to generate a list of weight matrices based on the number of neurons in each consecutive layers
generateWeightsMatrices :: [Int] -> Double -> Double -> [Matrix Double]
generateWeightsMatrices neurons a b = zipWith (\m n -> generateWeightsMatrix (m, n) a b) (tail neurons) neurons

-- * Data Types

-- | Data type representing an activation function in a neural network.
--   Contains the function, its derivative, and a description.
data ActivationFunc = ActivationFunc{
    -- | The activation function itself. 
    --   Takes a 'Double' and returns a 'Double'.
    aF :: Double -> Double,

    -- | The derivative of the activation function.
    --   This is necessary for backpropagation in neural networks.
    aF' :: Double -> Double,

    -- | A descriptive string for the activation function.
    desc :: String
}

-- | Data type representing a layer in a neural network.
--   Contains the weights matrix and the activation function used by the layer.
data Layer = Layer {
    -- | The weights matrix of the layer.
    --   It is a 'Matrix' of 'Double' values.
    lWeights :: Matrix Double,

    -- | The activation function used by this layer.
    lAF :: ActivationFunc
}

printLayers :: [Layer] -> IO()
printLayers [] = return ()
printLayers (l:ls) = do
    print (lWeights l)
    printLayers ls

-- | Data type representing a backpropagation neural network.
--   Contains a list of network layers and a learning rate.
data BackpropNet = BackpropNet
    {
        -- | List of layers in the neural network.
        --   Each layer is represented by a 'Layer' data type.
        layers :: [Layer],

        -- | The learning rate of the neural network.
        --   Used during the training process to update the weights.
        learningRate :: Double
    }

-- | Data type representing a layer during the propagation phase in a neural network.
--   Contains various properties required for forward and backward propagation.
data PropagatedLayer 
    = PropagatedLayer
        {
            -- | Input to the layer. Represented as a column vector.
            propIn :: ColumnVector Double,

            -- | Output from the layer. Also a column vector.
            propOut :: ColumnVector Double,

            -- | First derivative of the activation function for this layer.
            --   This is used in the backpropagation process.
            propF'a :: ColumnVector Double,

            -- | Weights for this layer represented as a matrix.
            propW :: Matrix Double,

            -- | Activation function used by this layer.
            propAF :: ActivationFunc
        }
    | PropagatedInputLayer -- | Special case for the input layer where the output is the same as the input.
        {
            -- | Output from the input layer, which is the same as its input.
            propOut :: ColumnVector Double
        }

-- | Represents a layer in the backpropagation phase of a neural network.
--   Contains all the necessary components for the backward pass of the training algorithm.
data BackpropagatedLayer
    = BackpropagatedLayer{
        -- | Partial derivative of the cost with respect to the z-value (weighted input) of this layer.
        bpDazzle :: ColumnVector Double,

        -- | Gradient of the error with respect to the output of this layer.
        bpErrGrad :: ColumnVector Double,

        -- | Value of the first derivative of the activation function for this layer.
        bpF'a :: ColumnVector Double,

        -- | Input to this layer.
        bpIn :: ColumnVector Double,

        -- | Output from this layer.
        bpOut :: ColumnVector Double,

        -- | Weights for this layer.
        bpW :: Matrix Double,

        -- | Activation function specification for this layer.
        bpAF :: ActivationFunc
    }

-- | Type alias for a column vector.
--   Represented as a matrix but typically used to denote vectors in matrix computations, 
--   especially in the context of neural networks.
type ColumnVector a = Matrix a

-- * Helper Function

-- | Checks if the dimensions of two matrices are compatible for multiplication.
--   Throws an error if dimensions are inconsistent.
--
--   - Matrix 1: The matrix whose number of rows should match the number of columns in Matrix 2.
--   - Matrix 2: This matrix is returned if the dimensions match with Matrix 1.
--   - Returns: The second matrix, if the dimensions are compatible.
dimensionMatch :: Matrix Double -- ^ Matrix 1 (to be checked for matching dimensions with Matrix 2).
               -> Matrix Double -- ^ Matrix 2 (returned if dimensions with Matrix 1 match).
               -> Matrix Double -- ^ The second matrix, if the dimensions match.
dimensionMatch m1 m2
    | (nrows m1) == (ncols m2) = m2
    | otherwise = error "Inconsistent dimensions in weight matrix"

-- | Validates the input vector for a neural network.
--
--   Checks if the input vector has appropriate dimensions and values 
--   (typically in the range [0, 1]) to be processed by the network.
validateInput :: BackpropNet         -- ^ The neural network for input validation.
              -> ColumnVector Double -- ^ The input vector to be validated.
              -> ColumnVector Double -- ^ The validated input vector; throws error if validation fails.
validateInput net input
    | validSize = input
    -- | validSize && validValues = input
    | otherwise = error $ "Inconsistent dimensions in input vector. Expected size: " ++ 
        show fLayerSize ++ 
        ", Actual size: " ++ 
        show inputSize
    where 
        -- Checks if the number of columns in the weight matrix of the first layer equals the number of rows in the input.
        validSize = fLayerSize == inputSize
        -- Ensures all input values are within the range [0, 1].
        --validValues = all (\x -> x >= 0 && x <= 1) (toList input)
        inputSize = nrows input
        fLayerSize = ncols ( lWeights ( head ( layers net ) ) )

-- | Calculates the Hadamard product (element-wise multiplication) of two matrices.
-- 
--   This function uses an unsafe method for element-wise multiplication, assuming that the dimensions of the matrices
--   have been validated prior to the function call. It's faster but requires caution: ensure that the matrix dimensions 
--   match before using this function.
hadamardProduct :: Num a 
                => Matrix a -- ^ The first matrix in the Hadamard product.
                -> Matrix a -- ^ The second matrix in the Hadamard product.
                -> Matrix a -- ^ The resulting matrix after computing the Hadamard product.
hadamardProduct m1 m2 = elementwiseUnsafe (*) m1 m2 -- Uses an unsafe element-wise operation for performance.
    
-- * Initialise Backpropagation Network

-- | Constructs a backpropagation neural network (BackpropNet) using specified parameters.
-- 
--   - Learning rate: The learning rate for the network.
--   - Weight matrices: A list of weight matrices for each layer in the network.
--   - Activation function: The activation function to be used by each neuron in the network.
--   - Returns: A constructed backpropagation neural network (BackpropNet).
--
--   This function ensures that the dimensions of the weight matrices are compatible and
--   creates each layer with the specified activation function.
buildBackpropNet :: Double            -- ^ Learning rate.
                 -> [Matrix Double]   -- ^ List of weight matrices for each layer.
                 -> ActivationFunc    -- ^ Activation function for each neuron.
                 -> BackpropNet       -- ^ Constructed backpropagation neural network.
buildBackpropNet lr ws af = BackpropNet { layers = ls, learningRate = lr}
    where 
        checkedWeights = scanl1 dimensionMatch ws  -- Ensures the compatibility of matrix dimensions.
        ls = map createLayer checkedWeights        -- Creates each layer with given weights and activation function.
        createLayer w = Layer { lWeights = w, lAF = af }

-- * Propagation

-- | Propagates the state of one layer to the next in a neural network.
--
--   This function takes the output of a given layer and uses it as the input for the next layer. 
--   It applies the activation function of the next layer to this input, and computes the necessary 
--   values for forward and backward propagation.
propagate :: PropagatedLayer  -- ^ The previous layer whose output will be used as input for the next layer.
          -> Layer            -- ^ The next layer to which the output is to be propagated. 
                             --   This layer's weights and activation function will be used.
          -> PropagatedLayer  -- ^ A new 'PropagatedLayer' representing the state of the next layer 
                             --   after the propagation, including its input, output, derivative of the 
                             --   activation function, weights, and activation function.
propagate layerJ layerK 
    = PropagatedLayer {
            propIn = x,  -- Input to layerK is the output of layerJ.
            propOut = y, -- Output after applying the activation function.
            propF'a = f'a, -- First derivative of the activation function.
            propW = w, -- Weights of layerK.
            propAF = lAF layerK -- Activation function of layerK.
        }
    where 
        x = propOut layerJ -- Output of layer J.
        w = lWeights layerK -- Weights from layer K.
        a = multStd w x -- Matrix multiplication wx, resulting in a column vector.
        f = aF ( lAF layerK ) -- Activation function of layerK.
        -- The function passed to fmap (mapCol) takes the current row as an additional argument.
        y = fmap f a 
        f' = aF' ( lAF layerK )
        f'a = fmap f' a

-- | Propagates an input vector through the neural network, returning the state of each layer after propagation.
--   
--   This function takes an input vector and a neural network, then computes the output for each layer.
--   It starts with a special input layer and then applies the 'propagate' function to each subsequent layer.
propagateNet :: ColumnVector Double -- ^ The input vector to the neural network.
             -> BackpropNet         -- ^ The backpropagation neural network through which the input is propagated.
             -> [PropagatedLayer]   -- ^ A list of 'PropagatedLayer' representing the state of each layer after the input is propagated through them.
propagateNet input backNet = tail propLayers
    where 
        propLayers = scanl propagate layer0 (layers backNet) -- Sequentially propagates through the network layers.
        layer0 = PropagatedInputLayer{ propOut = validateInputs } -- Initial layer to start the propagation.
        validateInputs = validateInput backNet input -- Validates the input vector before starting the propagation.

-- | Performs backpropagation for a single layer of a neural network.
--
--   This function computes various gradients and derivatives necessary for updating the neural network's weights 
--   during training. It essentially calculates the contribution of a layer to the overall error of the network.
backpropagate :: PropagatedLayer      -- ^ The forward propagated state of the current layer.
              -> BackpropagatedLayer  -- ^ The backpropagated state of the next layer.
              -> BackpropagatedLayer  -- ^ The backpropagated state of the current layer after computing necessary values.
backpropagate layerJ layerK
    = BackpropagatedLayer {
        bpDazzle = dazzleJ,
        bpErrGrad = errorGrad dazzleJ f'aJ (propIn layerJ),
        bpF'a = propF'a layerJ,
        bpIn = propIn layerJ,
        bpOut = propOut layerJ,
        bpW = propW layerJ,
        bpAF = propAF layerJ
    }
    where
        -- The error term (dazzle) for the current layer is computed using the error term of the next layer,
        -- the derivative of the activation function, and the weights of the next layer.
        dazzleJ = multStd wKT (hadamardProduct dazzleK f'aK)
        dazzleK = bpDazzle layerK
        wKT = transpose (bpW layerK)
        f'aK = bpF'a layerK
        f'aJ = propF'a layerJ

-- | Calculates the gradient of the error with respect to the weights of a layer.
--   
--   This is used in the backpropagation step to update the weights.
errorGrad :: ColumnVector Double -- ^ The error term (dazzle) for the layer.
          -> ColumnVector Double -- ^ The derivative of the activation function for the layer (f'a).
          -> ColumnVector Double -- ^ The input to the layer.
          -> Matrix Double       -- ^ The gradient of the error with respect to the layer's weights.
errorGrad dazzle f'a input = multStd (hadamardProduct dazzle f'a) (transpose input)

-- | Performs the backpropagation step for the final (output) layer of a neural network.
--
--   This function computes the initial error term and error gradient for the final layer, 
--   based on the difference between the network's output and the target output.
backpropagateFinalLayer :: PropagatedLayer      -- ^ The forward propagated state of the output layer.
                        -> ColumnVector Double  -- ^ The target output vector.
                        -> BackpropagatedLayer  -- ^ The backpropagated state of the output layer.
backpropagateFinalLayer layerK target
    = BackpropagatedLayer {
        bpDazzle = dazzle,
        bpErrGrad = errorGrad dazzle f'a (propIn layerK),
        bpF'a = propF'a layerK,
        bpIn = propIn layerK,
        bpOut = propOut layerK,
        bpW = propW layerK,
        bpAF = propAF layerK
    }
    where
        dazzle = elementwiseUnsafe (-) (propOut layerK) target
        f'a = propF'a layerK

-- | Executes backpropagation across all layers of a neural network.
--
--   This function performs backpropagation in reverse order, starting from the output layer.
backpropagateNet :: ColumnVector Double            -- ^ The target output vector for the entire network.
                 -> [PropagatedLayer]              -- ^ List of layers in their forward propagated state.
                 -> [BackpropagatedLayer]          -- ^ List of layers in their backpropagated state.
backpropagateNet target propLayers
    = scanr backpropagate outputLayer hiddenLayers
    where
        hiddenLayers = init propLayers
        outputLayer = backpropagateFinalLayer (last propLayers) target

-- | Updates the weights of a layer in a neural network based on the backpropagation results.
--
--   The function adjusts the layer's weights by a scaled error gradient, 
--   effectively 'learning' from the errors made in the forward pass.
update :: Double              -- ^ The learning rate, determining how much the weights are adjusted.
       -> BackpropagatedLayer -- ^ The backpropagated layer containing the error gradient and current weights.
       -> Layer               -- ^ The updated layer with new weights.
update rate layer = Layer { lWeights = wNew, lAF = bpAF layer}
    where
        delW = scaleMatrix rate (bpErrGrad layer) -- Scales the error gradient by the learning rate.
        wNew = elementwiseUnsafe (-) (bpW layer) delW -- Subtracts the scaled error gradient from the current weights.

updateLayers :: Double
             -> [BackpropagatedLayer]
             -> [Layer]
updateLayers _ [] = []
updateLayers rate (l:ls)
    = update rate l : updateLayers rate ls

-- * New functions

-- Function to initialize a neural network
initializeNetwork :: [Int] -> [ActivationFunc] -> Double -> Double -> Double -> BackpropNet
initializeNetwork neurons activationFuncs a b lRate =
    BackpropNet { layers = networkLayers, learningRate = lRate }
    where
        weightMatrices = generateWeightsMatrices neurons a b
        networkLayers = zipWith Layer weightMatrices activationFuncs

-- Training on a batch of data
trainBatch :: BackpropNet -> [(ColumnVector Double, ColumnVector Double)] -> BackpropNet
trainBatch net dataset = foldl trainSingleExample net dataset
    where
        trainSingleExample :: BackpropNet -> (ColumnVector Double, ColumnVector Double) -> BackpropNet
        trainSingleExample currentNet (input, target) = currentNet { layers = updatedLayers }
            where
                propagatedLayers = propagateNet input currentNet
                backpropagatedLayers = backpropagateNet target propagatedLayers
                updatedLayers = updateLayers (learningRate currentNet) backpropagatedLayers

-- | Trains on a batch of data and logs the training process to a specified file.
trainBatchLogged :: BackpropNet -> [(ColumnVector Double, ColumnVector Double)] -> FilePath -> IO BackpropNet
trainBatchLogged net dataset logFilePath = withFile logFilePath AppendMode $ \handle -> 
    foldM (trainSingleExample handle) net dataset
    where
        trainSingleExample :: Handle -> BackpropNet -> (ColumnVector Double, ColumnVector Double) -> IO BackpropNet
        trainSingleExample handle currentNet (input, target) = do
            hPutStrLn handle "Training on a new example..."
            let propagatedLayers = propagateNet input currentNet  -- Assuming this is a pure function; adapt if necessary
            let backpropagatedLayers = backpropagateNet target propagatedLayers
            let updatedLayers = updateLayers (learningRate currentNet) backpropagatedLayers
            hPutStrLn handle "Example training complete."
            return currentNet { layers = updatedLayers }



trainEpochs :: BackpropNet -> Int -> [(ColumnVector Double, ColumnVector Double)] -> BackpropNet
trainEpochs net 0 _ = net
trainEpochs net epochs dataset = trainEpochs trainedNet (epochs - 1) dataset
    where
        trainedNet = trainBatch net dataset

meanSquaredError :: ColumnVector Double -> ColumnVector Double -> Double
meanSquaredError output target = (sum . toList $ fmap (^(2 :: Int)) (elementwiseUnsafe (-) output target)) / fromIntegral (nrows output)


evaluate :: BackpropNet -> [(ColumnVector Double, ColumnVector Double)] -> Double
evaluate net dataset = totalLoss / fromIntegral (length dataset)
    where
        totalLoss = sum $ map (\(input, target) -> meanSquaredError (extractOutput (propagateNet input net)) target) dataset
        extractOutput propLayers = propOut (last propLayers)

getOutput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
getOutput net input = extractOutput (propagateNet input net)
    where
        extractOutput propLayers = propOut (last propLayers)

{-
saveModel :: BackpropNet -> FilePath -> IO ()
saveModel net path = writeFile path (show net)

loadModel :: FilePath -> IO BackpropNet
loadModel path = read <$> readFile path
-}

-- identity activation function
identityAF :: ActivationFunc
identityAF = ActivationFunc
     {
       aF = id,
       aF' = const 1.0,
       desc = "identity"
    }

tanhAF :: ActivationFunc
tanhAF = ActivationFunc
    {
        aF = \x -> (exp x - exp (-x)) / (exp x + exp (-x)),
        aF' = \x -> 1 - (tanh x) ^ (2 :: Int),
        desc = "Tanh"
    }

activationFuncType :: ActivationFunc -> String
activationFuncType af = desc af
