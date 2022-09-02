package ru.mralexeimk.cnn.models;

import org.junit.Test;
import ru.mralexeimk.cnn.enums.PaddingFill;

import static org.junit.Assert.*;

public class MatrixExtractorTest {
    @Test
    public void testSumOperation() {
        Matrix A = new Matrix("1,2,3|4,5,5|-1.5,0,5.123");
        Matrix B = new Matrix("2,3,4|0,1.5,2|4,2.2,0");
        assertEquals(new Matrix("3,5,7|4,6.5,7|2.5,2.2,5.123"), MatrixExtractor.getSum(A, B));
    }

    @Test
    public void testMultiplyOperation() {
        Matrix A = new Matrix("1,5,6|-1,0,0|2,3,4");
        Matrix B = new Matrix("1,2,2|0.5,5,5|-5,6,0");
        assertEquals(new Matrix("-26.5,63,27|-1,-2,-2|-16.5,43,19"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1,2,3");
        B = new Matrix("-2|5|6");
        assertEquals(new Matrix("26"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1|2|3");
        B = new Matrix("-4,5,1");
        assertEquals(new Matrix("-4,5,1|-8,10,2|-12,15,3"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1.25");
        B = new Matrix("4");
        assertEquals(new Matrix("5"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1,4,5|1.5,0,4|-1,5,5");
        B = new Matrix("2|5|-4");
        assertEquals(new Matrix("2|-13|3"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1.3456987654321");
        B = new Matrix("-0.234565432345");
        assertEquals(new Matrix("-0.31565441272"), MatrixExtractor.getMultiply(A, B));

        A = new Matrix("1,2|3,4|0.5,6");
        B = new Matrix("0,8|1,2|4,3");
        assertEquals(new Matrix("0,16|3,8|2,18"), MatrixExtractor.getMultiply(A, B));
    }

    @Test
    public void testTransposeOperation() {
        Matrix A = new Matrix(
                "2,3,4|" +
                       "-1,0,1|" +
                       "2,6,8");
        assertEquals(new Matrix("2,-1,2|3,0,6|4,1,8"), MatrixExtractor.getTransposed(A));

        A = new Matrix("5.523,65,-23456,75");
        assertEquals(new Matrix("5.523|65|-23456|75"), MatrixExtractor.getTransposed(A));
    }

    @Test
    public void testInverseOperation() {
        Matrix A = new Matrix(
                "2,3,4|" +
                        "-1,0,1|" +
                        "2,6,8");
        assertEquals(new Matrix("1,0,-0.5|-1.6667,-1.333,1|1,1,-0.5"), MatrixExtractor.getInverse(A));
    }

    @Test
    public void testConvolutionOperation() {
        Matrix A = new Matrix(3, 3, 1);
        Matrix K = new Matrix(2, 2, 1);
        assertEquals(new Matrix(2, 2, 4), MatrixExtractor.getConvertByKernel(A, K));

        A = new Matrix(5, 5, 1);
        K = new Matrix(2, 2, 1);
        assertEquals(new Matrix(4, 4, 4), MatrixExtractor.getConvertByKernel(A, K));

        A = new Matrix(7, 7, 1);
        K = new Matrix(7, 7, 1);
        assertEquals(new Matrix(1, 1, 49), MatrixExtractor.getConvertByKernel(A, K));

        A = new Matrix("1,2,3|1,2,3|1,2,3");
        K = new Matrix(2, 2, 1);
        assertEquals(new Matrix("6,10|6,10"), MatrixExtractor.getConvertByKernel(A, K));

        A = new Matrix("1,2,2,4|6,3,1,3|1,2,3,4|1,1,1,1");
        K = new Matrix("5,4|2,1");
        assertEquals(new Matrix("28,25,31|46,26,27|16,25,34"), MatrixExtractor.getConvertByKernel(A, K));

        A = new Matrix(3, 3, 1);
        K = new Matrix(2, 2, 1);
        assertEquals(new Matrix("1,2,2,1|2,4,4,2|2,4,4,2|1,2,2,1"), MatrixExtractor.getConvertByKernel(A, K, 1, PaddingFill.BY_ZEROES));

        A = new Matrix(3, 3, 1);
        K = new Matrix(2, 2, 1);
        assertEquals(new Matrix("0,0,0,0,0,0|0,1,2,2,1,0|0,2,4,4,2,0|0,2,4,4,2,0|0,1,2,2,1,0|0,0,0,0,0,0"),
                MatrixExtractor.getConvertByKernel(A, K, 2, PaddingFill.BY_ZEROES));

        A = new Matrix("4,2|2,3");
        K = new Matrix(2, 2, 2);
        assertEquals(new Matrix("0,0,0,0,0,0,0|0,0,0,0,0,0,0|0,0,8,12,4,0,0|" +
                        "0,0,12,22,10,0,0|0,0,4,10,6,0,0|0,0,0,0,0,0,0|0,0,0,0,0,0,0"),
                MatrixExtractor.getConvertByKernel(A, K, 3, PaddingFill.BY_ZEROES));

        A = new Matrix("1,1,1,1|2,2,2,2|3,3,3,3|4,4,4,4");
        K = new Matrix("2,2|1,1");
        assertEquals(new Matrix("8,8|20,20"), MatrixExtractor.getConvertByKernel(A, K, 0, PaddingFill.BY_ZEROES, 2));

        A = new Matrix("1,1,1,1|2,2,2,2|3,3,3,3|4,4,4,4");
        K = new Matrix("2,2|1,1");
        assertEquals(new Matrix("1,2|10,20"), MatrixExtractor.getConvertByKernel(A, K, 1, PaddingFill.BY_ZEROES, 3));
    }

    @Test
    public void testPaddingExpandOperation() {
        Matrix A = new Matrix("1,2|3,4");
        assertEquals(new Matrix("0,0,0,0|0,1,2,0|0,3,4,0|0,0,0,0"),
                MatrixExtractor.getPaddingExpandMatrix(A, 1, PaddingFill.BY_ZEROES));

        A = new Matrix("1");
        assertEquals(new Matrix("1,1,1|1,1,1|1,1,1"),
                MatrixExtractor.getPaddingExpandMatrix(A, 1, PaddingFill.BY_MEDIAN));

        A = new Matrix(1, 1, 2);
        assertEquals(new Matrix( "2,2,2,2,2|" +
                        "2,2,2,2,2|" +
                        "2,2,2,2,2|" +
                        "2,2,2,2,2|" +
                        "2,2,2,2,2"),
                MatrixExtractor.getPaddingExpandMatrix(A, 2, PaddingFill.BY_MEDIAN));
    }
}