import NN
import Data.Matrix

main :: IO()
main = do
    let neurons =  [20,10]  -- Example network structure: 2 input neurons, 2 hidden neurons, 2 output neurons
        activationFuncs = [tanhAF]  -- Activation functions for each layer
        a = -0.5  -- Lower bound for initial weights
        b = 0.5   -- Upper bound for initial weights
        lRate = 0.3  -- Learning rate
        
        -- Initialize the neural network
        initialNN = initializeNetwork neurons activationFuncs a b lRate

        -- Example input vector (2-dimensional for simplicity)
        inputVector = fromList 20 1 [1..20]

        -- Example target output
        targetOutput = fromList 10 1 [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]

        trainedNN = trainEpochs initialNN 20000 [(inputVector, targetOutput)] 

        score = evaluate trainedNN [(inputVector, targetOutput)]
    
    print $ ncols inputVector
    print $ nrows inputVector


    print score

    print $ getOutput initialNN inputVector

    print $ getOutput trainedNN inputVector

    