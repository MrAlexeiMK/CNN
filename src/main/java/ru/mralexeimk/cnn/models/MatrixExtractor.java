package ru.mralexeimk.cnn.models;

import ru.mralexeimk.cnn.enums.PaddingFill;
import ru.mralexeimk.cnn.other.Pair;

import java.util.ArrayList;
import java.util.List;

public class MatrixExtractor {

    /**
     * @return Sum of matrix and scalar
     */
    public static Matrix getSum(Matrix A, double val) {
        Matrix res = new Matrix(A.N, A.M);
        for(int x = 0; x < A.N; ++x) {
            for(int y = 0; y < A.M; ++y) {
                res.set(x, y, A.get(x, y) + val);
            }
        }
        return res;
    }

    /**
     * @return Diff of matrix and scalar
     */
    public static Matrix getDiff(Matrix A, double val) {
        return getSum(A, -val);
    }

    /**
     * @return Sum of two matrices (A+B)
     */
    public static Matrix getSum(Matrix A, Matrix B) {
        Matrix res = new Matrix(A.N, A.M);
        for(int x = 0; x < A.N; ++x) {
            for(int y = 0; y < A.M; ++y) {
                res.set(x, y, A.get(x, y) + B.get(x, y));
            }
        }
        return res;
    }

    /**
     * @return Diff of two matrices (A-B)
     */
    public static Matrix getDiff(Matrix A, Matrix B) {
        Matrix res = new Matrix(A.N, A.M);
        for(int x = 0; x < A.N; ++x) {
            for(int y = 0; y < A.M; ++y) {
                res.set(x, y, A.get(x, y) - B.get(x, y));
            }
        }
        return res;
    }

    /**
     * @return Multiply of matrix and scalar
     */
    public static Matrix getMultiply(Matrix A, double val) {
        Matrix res = new Matrix(A.N, A.M);
        for(int x = 0; x < A.N; ++x) {
            for(int y = 0; y < A.M; ++y) {
                res.set(x, y, A.get(x, y) * val);
            }
        }
        return res;
    }

    /**
     * @return Multiply of two matrices (A*B)
     */
    public static Matrix getMultiply(Matrix A, Matrix B) {
        if(A.N == B.getM()) {
            Matrix res = new Matrix(B.getN(), A.M);
            for (int y = 0; y < A.M; ++y) {
                for (int x = 0; x < B.getN(); ++x) {
                    for(int i = 0; i < A.N; ++i) {
                        res.set(x, y, res.get(x, y) + A.get(i, y)*B.get(x, i));
                    }
                }
            }
            return res;
        }
        else if(A.N == B.getN() && A.M == B.getM()) {
            Matrix res = new Matrix(A.N, A.M);
            for(int y = 0; y < A.M; ++y) {
                for(int x = 0; x < A.N; ++x) {
                    res.set(x, y, A.get(x, y) * B.get(x, y));
                }
            }
            return res;
        }
        throw new RuntimeException("Can't multiply matrices");
    }

    /**
     * @return Transposed matrix
     */
    public static Matrix getTransposed(Matrix A) {
        Matrix res = new Matrix(A.M, A.N);
        for (int x = 0; x < A.N; ++x) {
            for (int y = 0; y < A.M; ++y) {
                res.set(y, x, A.get(x, y));
            }
        }
        return res;
    }

    /**
     * @return Inverse matrix
     */
    public static Matrix getInverse(Matrix A) {
        if(A.N != A.M) throw new IndexOutOfBoundsException("Matrix must be square");
        Matrix E = new Matrix(A.N, A.M);
        E.setDiagonal(1);
        Matrix B = new Matrix(A);
        B.joinRight(E);
        B.toUnit();
        for(int i = 0; i < B.getM(); ++i) {
            if (B.get(i, i) == 0) throw new RuntimeException("Determinant mustn't be 0");
        }
        return B.getSubMatrix(A.N, 0, A.N, A.M);
    }

    /**
     * Convolutional operation on matrix A with kernel K
     * @param A Input matrix
     * @param K Kernel matrix
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K) {
        return getConvertByKernel(A, K, 0, 0, PaddingFill.BY_MEDIAN, 1, 1);
    }

    /**
     * Convolutional operation on matrix A with kernel K and other parameters
     * @param A Input matrix
     * @param K Kernel matrix
     * @param paddingSize Horizontal and Vertical padding
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K, int paddingSize) {
        return getConvertByKernel(A, K, paddingSize, paddingSize, PaddingFill.BY_MEDIAN, 1, 1);
    }

    /**
     * Convolutional operation on matrix A with kernel K and other parameters
     * @param A Input matrix
     * @param K Kernel matrix
     * @param paddingSize Horizontal and Vertical padding
     * @param paddingFill Padding Fill type
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K, int paddingSize, PaddingFill paddingFill) {
        return getConvertByKernel(A, K, paddingSize, paddingSize, paddingFill, 1, 1);
    }

    /**
     * Convolutional operation on matrix A with kernel K and other parameters
     * @param A Input matrix
     * @param K Kernel matrix
     * @param paddingSize Horizontal and Vertical padding
     * @param paddingFill Padding Fill type
     * @param stridingSize Horizontal and Vertical striding
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K, int paddingSize, PaddingFill paddingFill, int stridingSize) {
        return getConvertByKernel(A, K, paddingSize, paddingSize, paddingFill, stridingSize, stridingSize);
    }

    /**
     * Convolutional operation on matrix A with kernel K and other parameters
     * @param A Input matrix
     * @param K Kernel matrix
     * @param paddingSizeX Horizontal padding
     * @param paddingSizeY Vertical padding
     * @param paddingFill Padding Fill type
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K, int paddingSizeX, int paddingSizeY, PaddingFill paddingFill) {
        return getConvertByKernel(A, K, paddingSizeX, paddingSizeY, paddingFill, 1, 1);
    }

    /**
     * Convolutional operation on matrix A with kernel K and other parameters
     * @param A Input matrix
     * @param K Kernel matrix
     * @param paddingSizeX Horizontal padding
     * @param paddingSizeY Vertical padding
     * @param paddingFill Padding Fill type
     * @param stridingSizeX  Horizontal striding
     * @param stridingSizeY Vertical striding
     * @return Result of convolution
     */
    public static Matrix getConvertByKernel(Matrix A, Matrix K, int paddingSizeX, int paddingSizeY,
                                            PaddingFill paddingFill, int stridingSizeX, int stridingSizeY) {
        if(K.N > A.N || K.M > A.M) throw new RuntimeException("K.N > N or K.M > M");
        Matrix res = new Matrix(
                (A.N - K.N + 2 * paddingSizeY) / stridingSizeY + 1,
                (A.M - K.M + 2 * paddingSizeX) / stridingSizeX + 1
        );
        for (int y = 0; y < res.N; ++y) {
            for (int x = 0; x < res.M; ++x) {
                double val = 0;
                double sum = 0;
                List<Pair<Integer, Integer>> paddingZones = new ArrayList<>();
                for (int x1 = x * stridingSizeX - paddingSizeX;
                     x1 < x * stridingSizeX - paddingSizeX + K.M; ++x1) {
                    for (int y1 = y * stridingSizeY - paddingSizeY;
                         y1 < y * stridingSizeY - paddingSizeY + K.N; ++y1) {
                        if (x1 >= 0 && x1 < A.M && y1 >= 0 && y1 < A.N) {
                            val += A.get(x1, y1) * K.get(x1 - x * stridingSizeX + paddingSizeX,
                                    y1 - y * stridingSizeY + paddingSizeY);
                            sum += A.get(x1, y1);
                        }
                        else {
                            if (paddingFill == PaddingFill.BY_MEDIAN) paddingZones.add(new Pair<>(x1, y1));
                        }
                    }
                }
                sum /= (K.N * K.M - paddingZones.size());
                for (Pair<Integer, Integer> p : paddingZones) {
                    val += sum * K.get(p.getFirst() - x * stridingSizeX + paddingSizeX,
                            p.getSecond() - y * stridingSizeY + paddingSizeY);
                }
                res.set(x, y, val);
            }
        }
        return res;
    }

    /**
     * @return Resized matrix with scaling
     */
    public static Matrix getResized(Matrix A, int width, int height) {
        Matrix res = new Matrix(height, width);
        double scaleX = (double)A.M / width;
        double scaleY = (double)A.N / height;
        for (int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                res.set(x, y, A.get((int)(x*scaleX), (int)(y*scaleY)));
            }
        }
        return res;
    }

    /**
     * @return Resized matrix with scaling
     */
    public static Matrix getResized(Matrix A, int width) {
        return getResized(A, width, (A.N * width) / A.M);
    }

    /**
     * Expand matrix on each size
     * @param A Input Matrix
     * @param paddingSize Count of layers to expand
     * @return Expanded matrix
     */
    public static Matrix getPaddingExpandMatrix(Matrix A, int paddingSize) {
        return getPaddingExpandMatrix(A, paddingSize, PaddingFill.BY_MEDIAN, 3);
    }

    /**
     * Expand matrix on each size
     * @param A Input matrix
     * @param paddingSize Count of layers to expand
     * @param paddingFill Padding Fill type
     * @return Expanded matrix
     */
    public static Matrix getPaddingExpandMatrix(Matrix A, int paddingSize, PaddingFill paddingFill) {
        return getPaddingExpandMatrix(A, paddingSize, paddingFill, 3);
    }

    /**
     * Expand matrix on each size
     * @param A Input matrix
     * @param paddingSize Count of layers to expand
     * @param paddingFill Padding Fill type
     * @param window Average window radius (only for PaddingFill = BY_AVERAGE)
     * @return Expanded matrix
     */
    public static Matrix getPaddingExpandMatrix(Matrix A, int paddingSize, PaddingFill paddingFill, int window) {
        if (paddingSize <= 0) throw new RuntimeException("PaddingSize should be > 0");
        if (window % 2 == 0) throw new RuntimeException("Window size should be odd");
        Matrix res = new Matrix(A.N+2*paddingSize, A.M+2*paddingSize, 0);
        res.replace(paddingSize, paddingSize, A);
        if(paddingFill == PaddingFill.BY_MEDIAN) {
            while (paddingSize > 0) {
                int x = paddingSize - 1, y = paddingSize - 1;
                while (x <= res.M - paddingSize)  {
                    res.updateByAverageWindow(x, y, window);
                    ++x;
                }
                --x;
                ++y;
                while (y <= res.N - paddingSize)  {
                    res.updateByAverageWindow(x, y, window);
                    ++y;
                }
                --y;
                --x;
                while (x >= paddingSize - 1)  {
                    res.updateByAverageWindow(x, y, window);
                    --x;
                }
                ++x;
                --y;
                while (y >= paddingSize - 1)  {
                    res.updateByAverageWindow(x, y, window);
                    --y;
                }
                --paddingSize;
            }
        }
        return res;
    }

    /**
     * Scale matrix
     * @param A Input matrix
     * @param scale Scale value (scale < 1 - decrease, scale > 1 - increase)
     * @return Scaled matrix
     */
    public static Matrix getScale(Matrix A, double scale)  {
        if (scale <= 0) throw new RuntimeException("Scale should be > 0, your scale is "+scale);
        if (scale == 1) return A;
        if(scale < 1) {
            int size = (int)(1 / scale);
            Matrix K = new Matrix(size, size, 1.0 / (size * size));
            return getConvertByKernel(A, K, 0, PaddingFill.BY_ZEROES, size);
        }
        else {
            int sizeX = (int)(A.M * scale);
            int sizeY = (int)(A.N * scale);
            Matrix res = new Matrix(sizeY, sizeX);
            for(int y = 0; y < sizeY; ++y) {
                for(int x = 0; x < sizeX; ++x) {
                    res.set(x, y, A.get((int)(x/scale), (int)(y/scale)));
                }
            }
            return res;
        }
    }

    /**
     * Median Filter operation on matrix
     * @param A Input matrix
     * @param K Median Filter radius
     * @return Median Filtered matrix
     */
    public static Matrix getMedianFiltered(Matrix A, int K) {
        if (K % 2 == 0) ++K;
        Matrix res = new Matrix(A.N, A.M);
        Matrix I = getPaddingExpandMatrix(A, K/2);
        for(int x = 0; x < A.M; ++x)  {
            for(int y = 0; y < A.N; ++y) {
                List<Double> list = new ArrayList<>();
                for(int x_ = x - K/2; x_ <= x + K/2; ++x_) {
                    for (int y_ = y - K / 2; y_ <= y + K / 2; ++y_)  {
                        list.add(I.get(x_ + K/2, y_ + K/2));
                    }
                }

                int k = (K*K+1)/2;
                if (Math.abs(list.get(k) - I.get(x, y)) < Math.abs(list.get(K * K - k + 1)) - I.get(x, y)) {
                    res.set(x, y, list.get(k));
                }
                else res.set(x, y, list.get(K*K-k+1));
            }
        }
        return res;
    }

    /**
     * Define matrix by pattern string
     * @param pattern Pattern of string
     * @param separatorRows Rows separator
     * @param separatorColumns Columns separator
     */
    public static Matrix getMatrixFromExpression(String pattern, String separatorRows, String separatorColumns) {
        String[] arr = pattern.split(separatorRows);
        Matrix res = new Matrix(arr.length, arr[0].split(separatorColumns).length);
        for(int y = 0; y < arr.length; ++y) {
            String[] nums = arr[y].split(separatorColumns);
            for(int x = 0; x < nums.length; ++x) {
                res.set(x, y, Double.parseDouble(nums[x]));
            }
        }
        return res;
    }
}
