package ru.mralexeimk.cnn.models;

import org.junit.Test;

import static org.junit.Assert.*;

public class MatrixTest {
    @Test
    public void testReplaceOperation() {
        Matrix A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.replace(0, 0, new Matrix("1,0|0,1"));
        assertEquals(new Matrix("1,0,3|0,1,6|7,8,9"), A);

        A = new Matrix("1,2,3,1|4,5,6,1|7,8,9,1|1,1,1,1");
        A.replace(2, 2, new Matrix("0,0|0,0"));
        assertEquals(new Matrix("1,2,3,1|4,5,6,1|7,8,0,0|1,1,0,0"), A);
    }

    @Test
    public void testExpandOperation() {
        Matrix A = new Matrix("1,2|3,4");
        A.expand(1, 1);
        assertEquals(new Matrix("1,2,0|3,4,0|0,0,0"), A);

        A = new Matrix("1,2|3,4");
        A.expand(2, 0);
        assertEquals(new Matrix("1,2,0,0|3,4,0,0"), A);
    }

    @Test
    public void testIncreaseOperation() {
        Matrix A = new Matrix("1,2|3,4");
        A.increase(2, 2);
        assertEquals(new Matrix("1,1,2,2|1,1,2,2|3,3,4,4|3,3,4,4"), A);
    }

    @Test
    public void testEraseOperation() {
        Matrix A = new Matrix("1,2|3,4");
        A.erase(1, 0);
        assertEquals(new Matrix("1|3"), A);
    }

    @Test
    public void testRemoveOperation() {
        Matrix A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.removeColumn(2);
        assertEquals(new Matrix("1,2|4,5|7,8"), A);

        A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.removeRow(0);
        assertEquals(new Matrix("4,5,6|7,8,9"), A);

        A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.removeRows(0, 1);
        assertEquals(new Matrix("7,8,9"), A);

        A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.removeColumns(1, 2);
        assertEquals(new Matrix("1|4|7"), A);
    }

    @Test
    public void testSubMatrixOperation() {
        Matrix A = new Matrix("1,2,3|4,5,6|7,8,9");

        assertEquals(new Matrix("5,6|8,9"), A.getSubMatrix(1, 1, 2, 2));
        assertEquals(new Matrix("1,2,3|4,5,6|7,8,9"), A.getSubMatrix(0, 0, 3, 3));
    }

    @Test
    public void testToTriangularOperation() {
        Matrix A = new Matrix("1,2,3|6,4,1|1,9,8");
        A.toTriangularDown();
        assertEquals(new Matrix("1,2,3|0,-8,-17|0,0,-9.875"), A);

        A = new Matrix("1,2,3|6,4,1|1,9,8");
        A.toTriangularUp();
        assertEquals(new Matrix("1,0,0|6,4,0|1,9,8"), A);
    }

    @Test
    public void testJoinOperator() {
        Matrix A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.joinRight(new Matrix("1|2|3"));
        assertEquals(new Matrix("1,2,3,1|4,5,6,2|7,8,9,3"), A);

        A = new Matrix("1,2,3|4,5,6|7,8,9");
        A.joinBottom(new Matrix("1,2,3"));
        assertEquals(new Matrix("1,2,3|4,5,6|7,8,9|1,2,3"), A);
    }
}