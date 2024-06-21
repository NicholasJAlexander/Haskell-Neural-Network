module ActivationFunctions (
    ActivationFunc(..),
    identityAF,
    tanhAF,
    sigmoidAF,
    reluAF,
    leakyReluAF,
    eluAF
) where


-- | Data type representing an activation function in a neural network.
--   Contains the function, its derivative, and a description.
data ActivationFunc = ActivationFunc {
    -- | The activation function itself. 
    --   Takes a 'Double' and returns a 'Double'.
    aF :: Double -> Double,

    -- | The derivative of the activation function.
    --   This is necessary for backpropagation in neural networks.
    aF' :: Double -> Double,

    -- | A descriptive string for the activation function.
    desc :: String
}

-- Identity activation function
identityAF :: ActivationFunc
identityAF = ActivationFunc {
    aF = id,
    aF' = const 1.0,
    desc = "identity"
}

-- Tanh activation function
tanhAF :: ActivationFunc
tanhAF = ActivationFunc {
    aF = \x -> (exp x - exp (-x)) / (exp x + exp (-x)),
    aF' = \x -> 1 - (tanh x) ^ (2 :: Int),
    desc = "Tanh"
}

-- Sigmoid activation function
sigmoidAF :: ActivationFunc
sigmoidAF = ActivationFunc {
    aF = \x -> 1 / (1 + exp (-x)),  -- The sigmoid function
    aF' = \x -> let sig = 1 / (1 + exp (-x))
                in sig * (1 - sig),  -- The derivative of the sigmoid function
    desc = "Sigmoid"
}

-- ReLU activation function
reluAF :: ActivationFunc
reluAF = ActivationFunc {
    aF = \x -> max 0 x,
    aF' = \x -> if x > 0 then 1 else 0,
    desc = "ReLU"
}

-- Leaky ReLU activation function
leakyReluAF :: ActivationFunc
leakyReluAF = ActivationFunc {
    aF = \x -> if x > 0 then x else 0.01 * x,
    aF' = \x -> if x > 0 then 1 else 0.01,
    desc = "Leaky ReLU"
}

-- ELU activation function
eluAF :: ActivationFunc
eluAF = ActivationFunc {
    aF = \x -> if x >= 0 then x else (exp x - 1),
    aF' = \x -> if x >= 0 then 1 else exp x,
    desc = "ELU"
}
