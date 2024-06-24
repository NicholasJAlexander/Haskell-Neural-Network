module Main where

import Matrix
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV
import Control.Exception (catch, evaluate, SomeException)
import Codec.Picture.Metadata (Value(Double))

main :: IO ()
main = do
    -- Test fromLists function
    putStrLn "Testing fromLists function:"
    let validMatrix :: [[Double]]
        validMatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let invalidMatrix :: [[Double]]
        invalidMatrix = [[1, 2, 3], [4, 5], [6, 7, 8]]

    let matrix1 :: Matrix Double
        matrix1 = fromLists validMatrix
    putStrLn "Matrix 1:"
    printMatrix matrix1

    putStrLn "Attempting to create an invalid matrix:"
    catch (evaluate (fromLists invalidMatrix) >> putStrLn "Failed to catch error for invalid matrix")
          (\e -> putStrLn $ "Caught expected error: " ++ show (e :: SomeException))

    -- Test ncols and nrows functions
    putStrLn "Testing ncols and nrows functions:"
    let cols = ncols matrix1
    let rows = nrows matrix1
    putStrLn $ "Columns: " ++ show cols
    putStrLn $ "Rows: " ++ show rows

    -- Test hadamardProduct function
    putStrLn "Testing hadamardProduct function:"
    let matrix2 = fromLists [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    let hadamard = hadamardProduct matrix1 matrix2
    putStrLn "Hadamard Product of Matrix 1 and Matrix 2:"
    printMatrix hadamard

    -- Test scaleMatrix function
    putStrLn "Testing scaleMatrix function:"
    let scaledMatrix = scaleMatrix 2 matrix1
    putStrLn "Matrix 1 scaled by 2:"
    printMatrix scaledMatrix

    -- Test elementwise function
    putStrLn "Testing elementwiseUnsafe function with addition:"
    let elementwiseSum = elementwise (+) matrix1 matrix2
    putStrLn "Element-wise sum of Matrix 1 and Matrix 2:"
    printMatrix elementwiseSum


    -- Test multStd function
    putStrLn "Testing multStd function:"

    let matrixA :: Matrix Double
        matrixA = fromLists [[1, 2, 3], [4, 5, 6]]
    let matrixB :: Matrix Double
        matrixB = fromLists [[7, 8], [9, 10], [11, 12]]

    putStrLn "Matrix A:"
    printMatrix matrixA

    putStrLn "Matrix B:"
    printMatrix matrixB

    let productMatrix = multStd matrixA matrixB 
    putStrLn "Product of Matrix A and Matrix B:"
    printMatrix productMatrix

    let matrixC = fromLists [[1, 2], [3, 4]]

    putStrLn "Attempting to multiply incompatible matrices (Matrix A and Matrix C):"
    catch (evaluate (multStd matrixA matrixC) >> putStrLn "Failed to catch error for incompatible matrices")
        (\e -> putStrLn $ "Caught expected error: " ++ show (e :: SomeException))



    -- Test errorGrad function
    putStrLn "Testing errorGrad function:"

    let dazzle :: ColumnVector Double
        dazzle = UV.fromList [0.1, 0.2, 0.3]
    let f'a :: ColumnVector Double
        f'a = UV.fromList [0.5, 0.6, 0.7]
    let input :: ColumnVector Double 
        input = UV.fromList [1.0, 2.0, 3.0]

    let expectedGradient = fromLists [
            [0.05, 0.10, 0.15000000000000002],
            [0.12, 0.24, 0.36],
            [0.21, 0.42, 0.63]]

    putStrLn "Dazzle vector:"
    print dazzle

    putStrLn "f'a vector:"
    print f'a

    putStrLn "Input vector:"
    print input

    let computedGradient = errorGrad dazzle f'a input

    putStrLn "Computed Gradient Matrix:"
    printMatrix computedGradient

    putStrLn "Expected Gradient Matrix:"
    printMatrix expectedGradient

    let result = if computedGradient == expectedGradient
                then "Test passed!"
                else "Test failed!"
    putStrLn result



    -- Test matrixToVector function
    putStrLn "Testing matrixToVector function:"

    let singleColumnMatrix :: Matrix Double
        singleColumnMatrix = fromLists [[1.0], [2.0], [3.0]]
    let expectedVector :: ColumnVector Double
        expectedVector = UV.fromList [1.0, 2.0, 3.0]

    putStrLn "Single-column Matrix:"
    printMatrix singleColumnMatrix

    let convertedVector = matrixToVector singleColumnMatrix

    putStrLn "Converted Vector:"
    print convertedVector

    putStrLn "Expected Vector:"
    print expectedVector

    let result = if convertedVector == expectedVector
                then "Test passed!"
                else "Test failed!"
    putStrLn result



    -- Test transpose function
    putStrLn "Testing transpose function:"

    let originalMatrix :: Matrix Double
        originalMatrix = fromLists [[1, 2, 3], [4, 5, 6]]
    let expectedTransposedMatrix :: Matrix Double
        expectedTransposedMatrix = fromLists [[1, 4], [2, 5], [3, 6]]

    putStrLn "Original Matrix:"
    printMatrix originalMatrix

    let transposedMatrix = transpose originalMatrix

    putStrLn "Transposed Matrix:"
    printMatrix transposedMatrix

    putStrLn "Expected Transposed Matrix:"
    printMatrix expectedTransposedMatrix

    let resultTranspose = if transposedMatrix == expectedTransposedMatrix
                        then "Test passed!"
                        else "Test failed!"
    putStrLn resultTranspose

    -- Test vectorToMatrix function
    putStrLn "Testing vectorToMatrix function:"

    let vector :: ColumnVector Double
        vector = UV.fromList [1.0, 2.0, 3.0]
    let expectedSingleColumnMatrix :: Matrix Double
        expectedSingleColumnMatrix = fromLists [[1.0], [2.0], [3.0]]

    putStrLn "Vector:"
    print vector

    let singleColumnMatrix = vectorToMatrix vector

    putStrLn "Single-column Matrix from Vector:"
    printMatrix singleColumnMatrix

    putStrLn "Expected Single-column Matrix:"
    printMatrix expectedSingleColumnMatrix

    let resultVectorToMatrix = if singleColumnMatrix == expectedSingleColumnMatrix
                            then "Test passed!"
                            else "Test failed!"
    putStrLn resultVectorToMatrix
