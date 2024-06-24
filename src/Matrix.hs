module Matrix where
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV


-- | Create a vector of vectors to represent a matrix
type Matrix a = V.Vector (UV.Vector a)

type ColumnVector a = UV.Vector a

-- * Matrix Functions

-- | Converts a list of lists to a Vector of Vectors
-- This function now checks for uniformity of inner list lengths to ensure a valid matrix.
fromLists :: UV.Unbox a => [[a]] -> Matrix a
fromLists xss
    | all (== head lengths) lengths = V.fromList . map UV.fromList $ xss
    | otherwise = error "All rows must have the same number of columns"
  where lengths = map length xss

-- | Returns number of Columns in matrix
ncols :: UV.Unbox a => Matrix a -> Int
ncols m = if V.null m then 0 else UV.length (V.head m)

-- | Returns number of Rows in matrix
nrows :: UV.Unbox a => Matrix a -> Int
nrows = V.length

-- | Calculates the Hadamard product (element-wise multiplication) of two matrices.
-- Added check to ensure matrices are of the same dimensions.
hadamardProduct :: (Floating a, UV.Unbox a) => Matrix a -> Matrix a -> Matrix a
hadamardProduct m1 m2
    | nrows m1 /= nrows m2 || ncols m1 /= ncols m2 = error "Matrices must have the same dimensions for Hadamard product"
    | otherwise = V.zipWith (UV.zipWith (*)) m1 m2

-- Print a Matrix
printMatrix :: (Show a, UV.Unbox a) => Matrix a -> IO ()
printMatrix matrix = do
    mapM_ printRow (V.toList matrix)
    putStrLn ""  -- Add an extra newline at the end
  where
    printRow row = print (UV.toList row)

-- | Scales a matrix by a scalar
scaleMatrix :: (Num a, UV.Unbox a) => a -> Matrix a -> Matrix a
scaleMatrix scale = V.map (UV.map (* scale))

-- Function to perform element-wise operation on two matrices
-- Checks that both matrices have the same dimensions before applying the function.
elementwise :: (UV.Unbox a) => (a -> a -> a) -> Matrix a -> Matrix a -> Matrix a
elementwise f m1 m2
    | nrows m1 /= nrows m2 || ncols m1 /= ncols m2 = error "Matrices must have the same dimensions"
    | otherwise = V.zipWith (UV.zipWith f) m1 m2


-- | Standard matrix multiplication.
-- Multiplies two matrices using standard matrix multiplication algorithm.
multStd :: (Num a, UV.Unbox a) => Matrix a -> Matrix a -> Matrix a
multStd m1 m2
    | ncols m1 /= nrows m2 = error "Number of columns in the first matrix must equal number of rows in the second"
    | otherwise = V.generate (nrows m1) $ \i ->
        UV.generate (ncols m2) $ \j ->
            UV.sum $ UV.zipWith (*) (m1 V.! i) (UV.generate (nrows m2) $ \k -> (m2 V.! k) UV.! j)

-- | Transpose a matrix.
-- Converts rows of the original matrix into columns of the new matrix.
transpose :: UV.Unbox a => Matrix a -> Matrix a
transpose m
    | V.null m = V.empty
    | otherwise = V.generate (ncols m) $ \i ->
        UV.generate (nrows m) $ \j -> (m V.! j) UV.! i

-- | Converts a column vector into a single-column matrix where each element of the vector
-- | becomes a separate row in the matrix.
vectorToMatrix :: UV.Unbox a => UV.Vector a -> Matrix a
vectorToMatrix v = V.generate (UV.length v) (\i -> UV.singleton (v UV.! i))


-- | This is used in the backpropagation step to update the weights.
errorGrad :: UV.Vector Double -- ^ The error term (dazzle) for the layer.
          -> UV.Vector Double -- ^ The derivative of the activation function for the layer (f'a).
          -> UV.Vector Double -- ^ The input to the layer.
          -> Matrix Double    -- ^ The gradient of the error with respect to the layer's weights.
errorGrad dazzle f'a input =
    multStd (vectorToMatrix (UV.zipWith (*) dazzle f'a)) (transpose (vectorToMatrix input))

-- | Converts a single-column matrix back into a column vector.
-- Assumes the matrix has exactly one column.
matrixToVector :: UV.Unbox a => Matrix a -> UV.Vector a
matrixToVector m = UV.concat $ V.toList m
