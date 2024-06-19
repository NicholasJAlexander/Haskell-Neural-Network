module Main where

import NN
import Data.Matrix
import System.Environment
import System.IO
import qualified Data.ByteString as B
import Data.Word (Word8)

-- Function to convert a 28x28 matrix to a 784x1 column vector
flattenMatrix :: Matrix Double -> Matrix Double
flattenMatrix mat = matrix 784 1 (\(i, _) -> mat ! (1 + (i - 1) `div` 28, 1 + (i - 1) `mod` 28))

-- Helper functions for loading MNIST data
loadMNISTImages :: FilePath -> IO [Matrix Double]
loadMNISTImages path = do
    contents <- B.readFile path
    let images = decodeIDX3 contents
    return $ map (flattenMatrix . fromLists . map (map (\x -> fromIntegral x / 255.0)) . chunksOf 28) images

loadMNISTLabels :: FilePath -> IO [Int]
loadMNISTLabels path = do
    contents <- B.readFile path
    let labels = decodeIDX1 contents
    return $ map fromIntegral (B.unpack labels)

decodeIDX1 :: B.ByteString -> B.ByteString
decodeIDX1 = B.drop 8

decodeIDX3 :: B.ByteString -> [[Word8]]
decodeIDX3 bs =
    let (_, rest) = B.splitAt 4 bs
        (numImagesBS, rest2) = B.splitAt 4 rest
        (_, imageData) = B.splitAt 8 rest2
        numImages = fromIntegral $ B.foldl' (\acc x -> acc * 256 + fromIntegral x) 0 numImagesBS
    in chunksOf (28 * 28) (B.unpack imageData)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs
  | n > 0 = take n xs : chunksOf n (drop n xs)
  | otherwise = error "Chunk size must be greater than zero"


labelToColumnVector :: Int -> Matrix Double
labelToColumnVector label = matrix 10 1 $ \(i, _) -> if i == label + 1 then 1.0 else 0.0

main :: IO ()
main = do
    -- Load the MNIST dataset
    trainImages <- loadMNISTImages "./examples/MNIST/train-images-idx3-ubyte"
    trainLabels <- loadMNISTLabels "./examples/MNIST/train-labels-idx1-ubyte"
    testImages <- loadMNISTImages "./examples/MNIST/t10k-images-idx3-ubyte"
    testLabels <- loadMNISTLabels "./examples/MNIST/t10k-labels-idx1-ubyte"

    -- Initialize the neural network
    let neurons = [784, 28, 10]
        activationFuncs = [tanhAF, tanhAF]
        learningRate = 0.1
        net = initializeNetwork neurons activationFuncs (-0.5) 0.5 learningRate

    -- Train the network
    let epochs = 1
        trainData = zip trainImages (map labelToColumnVector trainLabels)
    
    trainedNet <- trainBatchLogged net trainData "output.txt"

    printLayers (layers trainedNet)

    -- Evaluate the network on the test set
    {-
    let testData = zip testImages (map labelToColumnVector testLabels)
        accuracy = evaluate trainedNet testData
    putStrLn $ "Test accuracy: " ++ show accuracy
    -}