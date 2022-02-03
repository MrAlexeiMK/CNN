package ru.mralexeimk.cnn.models;

import ru.mralexeimk.cnn.other.Direction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Matrix3D implements Serializable {
    private List<Matrix> matrices;
    private int N, M, D;

    public Matrix3D(int N, int M, int D) {
        this.N = N;
        this.M = M;
        this.D = D;
        matrices = new ArrayList<>();
        for(int i = 0; i < D; ++i) {
            matrices.add(new Matrix(N, M));
        }
    }

    public Matrix3D(int N, int M, int D, List<Double> vector) {
        this(N, M, D);
        int k = 0;
        for(int z = 0; z < D; ++z) {
            for(int y = 0; y < M; ++y) {
                for(int x = 0; x < N; ++x) {
                    set(x, y, z, vector.get(k));
                    ++k;
                }
            }
        }
    }

    public Matrix3D(List<Double> vector, Direction direction) {
        N = 1;
        M = 1;
        D = 1;
        if(direction == Direction.X) N = vector.size();
        else if(direction == Direction.Y) M = vector.size();
        else D = vector.size();
        matrices = new ArrayList<>();
        for(int i = 0; i < D; ++i) {
            matrices.add(new Matrix(N, M));
        }
        int k = 0;
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                for(int d = 0; d < D; ++d) {
                    set(x, y, d, vector.get(k));
                    ++k;
                }
            }
        }
    }

    public Matrix3D(Matrix... ms) {
        N = ms[0].getN();
        M = ms[0].getM();
        D = ms.length;
        matrices = new ArrayList<>();
        for(int i = 0; i < D; ++i) {
            if(N == ms[i].getN() && M == ms[i].getM()) {
                matrices.add(ms[i].clone());
            }
            else {
                throw new IndexOutOfBoundsException("Matrices are different!");
            }
        }
    }

    public Matrix3D(List<Matrix> ms) {
        N = ms.get(0).getN();
        M = ms.get(0).getM();
        D = ms.size();
        matrices = new ArrayList<>();
        for(int i = 0; i < D; ++i) {
            if(N == ms.get(i).getN() && M == ms.get(i).getM()) {
                matrices.add(ms.get(i).clone());
            }
            else {
                throw new IndexOutOfBoundsException("Matrices are different!");
            }
        }
    }

    public Matrix3D(Matrix3D m) {
        assign(m);
    }

    public void assign(Matrix3D m) {
        N = m.getN();
        M = m.getM();
        D = m.getD();
        matrices = new ArrayList<>();
        for(int i = 0; i < D; ++i) {
            matrices.add(m.get(i).clone());
        }
    }

    public void redefine() {
        N = get(0).getN();
        M = get(0).getM();
        D = matrices.size();
    }

    public List<Double> toList() {
        List<Double> res = new ArrayList<>();
        for(int x = 0; x < N; ++x) {
            for(int y = 0; y < M; ++y) {
                for(int d = 0; d < D; ++d) {
                    res.add(get(x, y, d));
                }
            }
        }
        return res;
    }

    public boolean isVector() {
        return D == 1 && getMatrix().isVector();
    }

    public boolean isTransposeVector() {
        return D == 1 && getMatrix().isTransposeVector();
    }

    public Matrix getCloneMatrix() {
        return get(0).clone();
    }

    public Matrix getMatrix() {
        return get(0);
    }

    public Matrix3D resize(int N, int M, int D) {
        if(isVector() && this.M == N*M*D) {
            Matrix3D res = new Matrix3D(N, M, D);
            int k = 0;
            for (int z = 0; z < D; ++z) {
                for (int y = 0; y < M; ++y) {
                    for (int x = 0; x < N; ++x) {
                        res.set(x, y, z, get(0, k, 0));
                        ++k;
                    }
                }
            }
            assign(res);
        }
        return this;
    }

    public void saveExpand(int kX, int kY) {
        for(int z = 0; z < D; ++z) {
            get(z).saveExpand(kX, kY);
        }
        redefine();
    }

    public Matrix3D clone() {
        return new Matrix3D(this);
    }

    public int getN() {
        return N;
    }

    public int getM() {
        return M;
    }

    public int getD() {
        return D;
    }

    public Matrix get(int index) {
        return matrices.get(index);
    }

    public double get(int x, int y, int d) {
        return get(d).get(x, y);
    }

    public void set(int index, Matrix m) {
        if(N == m.getN() && M == m.getM()) {
            matrices.set(index, m);
        }
    }

    public void set(int x, int y, int d, double value) {
        matrices.get(d).set(x, y, value);
    }

    public void add(Matrix m) {
        ++D;
        matrices.add(m);
    }

    public Matrix3D padding(int size) {
        Matrix3D res = new Matrix3D(N+2*size, M+2*size, D);
        res.replace(size, size, this);
        assign(res);
        return this;
    }

    public void replace(int xStart, int yStart, Matrix3D m) {
        for(int i = 0; i < D; ++i) {
            get(i).replace(xStart, yStart, m.get(i));
        }
    }

    public Matrix3D convertByMaxPulling(int size) {
        N /= size;
        M /= size;
        for(int i = 0; i < D; ++i) {
            get(i).convertByMaxPulling(size);
        }
        return this;
    }

    public Matrix3D convertByAveragePulling(int size) {
        N /= size;
        M /= size;
        for(int i = 0; i < D; ++i) {
            get(i).convertByAveragePulling(size);
        }
        return this;
    }

    public Matrix3D convertByMinPulling(int size) {
        N /= size;
        M /= size;
        for(int i = 0; i < D; ++i) {
            get(i).convertByMinPulling(size);
        }
        return this;
    }

    public Matrix3D getConvertByAveragePulling(int size) {
        return new Matrix3D(this).convertByAveragePulling(size);
    }

    public Matrix3D getConvertByMaxPulling(int size) {
        return new Matrix3D(this).convertByMaxPulling(size);
    }

    public Matrix3D getConvertByMinPulling(int size) {
        return new Matrix3D(this).convertByMinPulling(size);
    }

    public Matrix3D getConvertByMergeKernel(Matrix3D K, int step) {
        int sizeX = (int)Math.ceil((N - K.getN() + 1)/(float)step);
        int sizeY = (int)Math.ceil((M - K.getM() + 1)/(float)step);
        Matrix3D res = new Matrix3D(sizeX, sizeY, K.getD());
        for(int i = 0; i < K.getD(); i+=step) {
            Matrix m = new Matrix(sizeX, sizeY);
            for (int j = 0; j < D; j+=step) {
                Matrix clone = get(j).clone();
                m.sum(clone.convertByKernel(K.get(i), step));
            }
            res.set(i, m);
        }
        return res;
    }

    public Matrix3D convertByMergeKernel(Matrix3D K, int step) {
        assign(getConvertByMergeKernel(K, step));
        return this;
    }

    public Matrix3D getConvertByKernel(Matrix3D K, int step) {
        int sizeX = (int)Math.ceil((N - K.getN() + 1)/(float)step);
        int sizeY = (int)Math.ceil((M - K.getM() + 1)/(float)step);
        Matrix3D res = new Matrix3D(sizeX, sizeY, K.getD());
        int len = K.getD()/D;
        for(int i = 0; i < K.getD(); ++i) {
            Matrix clone = get(i/len).clone();
            clone.convertByKernel(K.get(i), step);
            res.set(i, clone);
        }
        return res;
    }

    public Matrix3D convertByKernel(Matrix3D K, int step) {
        assign(getConvertByKernel(K, step));
        return this;
    }

    public Matrix3D getConvertToLine() {
        Matrix3D res = new Matrix3D(1, N*M*D, 1);
        for(int i = 0; i < D; ++i) {
            List<Double> list = get(i).toList();
            for(int j = 0; j < list.size(); ++j) {
                res.get(0).set(0, i*list.size()+j, list.get(j));
            }
        }
        return res;
    }

    public Matrix3D getConvertTo3DLine() {
        Matrix3D res = new Matrix3D(1, 1, N*M*D);
        for(int i = 0; i < D; ++i) {
            List<Double> list = get(i).toList();
            for(int j = 0; j < list.size(); ++j) {
                res.get(i*list.size()+j).set(0, 0, list.get(j));
            }
        }
        return res;
    }

    public Matrix3D getConvertByMultiply(Matrix3D data) {
        return new Matrix3D(get(0).getMultiply(data.get(0)));
    }

    public String toString() {
        String res = "("+N+";"+M+";"+D+")\n";
        for(int i = 0; i < D; ++i) {
            res += get(i);
        }
        return res;
    }

    public Matrix3D sum(double x) {
        for(int i = 0; i < D; ++i) {
            get(i).sum(x);
        }
        return this;
    }

    public Matrix3D sum(List<Double> vector) {
        for(int i = 0; i < D; ++i) {
            get(i).sum(vector.get(i));
        }
        return this;
    }

    public String getShapes() {
        return "("+N+";"+M+";"+D+")";
    }

    public void print() {
        System.out.println(this);
    }

    public void printShapes() {
        System.out.println(getShapes());
    }
}
