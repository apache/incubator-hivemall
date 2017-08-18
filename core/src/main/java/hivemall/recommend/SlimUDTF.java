package hivemall.recommend;

import hivemall.UDTFWithOptions;
import hivemall.math.matrix.sparse.DoKMatrix;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.*;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;


public class SlimUDTF extends UDTFWithOptions {
    private double l1;
    private double l2;
    private int iter;

    private DoKMatrix W = new DoKMatrix();

    private PrimitiveObjectInspector itemIOI;
    private PrimitiveObjectInspector itemJOI;
    private MapObjectInspector itemIRatesOI;
    private MapObjectInspector itemJRatesOI;

    private MapObjectInspector topKRatesOfIOI;
    private PrimitiveObjectInspector topKRatesOfIKeyOI;
    private MapObjectInspector topKRatesOfIValueOI;
    private PrimitiveObjectInspector topKRatesOfIValueKeyOI;
    private PrimitiveObjectInspector topKRatesOfIValueValueOI;

    private PrimitiveObjectInspector itemIRateKeyOI;
    private PrimitiveObjectInspector itemIRateValueOI;

    private PrimitiveObjectInspector itemJRateKeyOI;
    private PrimitiveObjectInspector itemJRateValueOI;

    private double loss;

    public SlimUDTF() {}

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;
        if (numArgs != 5 && numArgs != 6) {
            throw new UDFArgumentException(
                "_FUNC_ takes arguments: int i, map<int, double> r_i, map<int, map<int, double>> topKRatessOfI, int j, map<int, double> r_j, [, constant string options]");
        }

        this.itemIOI = HiveUtils.asIntCompatibleOI(argOIs[0]);

        this.itemIRatesOI = HiveUtils.asMapOI(argOIs[1]);
        this.itemIRateKeyOI = HiveUtils.asIntCompatibleOI((this.itemIRatesOI.getMapKeyObjectInspector()));
        this.itemIRateValueOI = HiveUtils.asDoubleCompatibleOI((this.itemIRatesOI.getMapValueObjectInspector()));

        this.topKRatesOfIOI = HiveUtils.asMapOI(argOIs[2]);
        this.topKRatesOfIKeyOI = HiveUtils.asIntCompatibleOI(topKRatesOfIOI.getMapKeyObjectInspector());
        this.topKRatesOfIValueOI = HiveUtils.asMapOI(topKRatesOfIOI.getMapValueObjectInspector());
        this.topKRatesOfIValueKeyOI = HiveUtils.asIntCompatibleOI(topKRatesOfIValueOI.getMapKeyObjectInspector());
        this.topKRatesOfIValueValueOI = HiveUtils.asDoubleCompatibleOI(topKRatesOfIValueOI.getMapValueObjectInspector());

        this.itemJOI = HiveUtils.asIntCompatibleOI(argOIs[3]);

        this.itemJRatesOI = HiveUtils.asMapOI(argOIs[4]);
        this.itemJRateKeyOI = HiveUtils.asIntCompatibleOI((this.itemJRatesOI.getMapKeyObjectInspector()));
        this.itemJRateValueOI = HiveUtils.asDoubleCompatibleOI((this.itemJRatesOI.getMapValueObjectInspector()));

        processOptions(argOIs);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("i");
        fieldNames.add("j");
        fieldNames.add("wij");

        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("l1", "l1coefficient", true,
            "Coefficient for l1 regularizer [default: 0.01]");
        opts.addOption("l2", "l2coefficient", true,
            "Coefficient for l2 regularizer [default: 0.01]");
        opts.addOption("iter", "iteration", true,
            "The number of iterations for coordinate descent [default: 40]");
        opts.addOption("disable_cv", "disable_cvtest", false,
            "Whether to disable convergence check [default: OFF]");
        opts.addOption("cv_rate", "convergence_rate", true,
            "Threshold to determine convergence [default: 0.005]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        if (argOIs.length >= 6) {
            String rawArgs = HiveUtils.getConstString(argOIs[5]);
            cl = parseOptions(rawArgs);

            this.l1 = Primitives.parseDouble(cl.getOptionValue("l1"), 0.01d);
            if (this.l1 < 0.d || this.l1 > 1.d) {
                throw new UDFArgumentException("Argument `double l1` must be within [0., 1.]: "
                        + this.l1);
            }

            this.l2 = Primitives.parseDouble(cl.getOptionValue("l2"), 0.01d);
            if (this.l2 < 0.d || this.l2 > 1.d) {
                throw new UDFArgumentException("Argument `double l2` must be within [0., 1.]: "
                        + this.l2);
            }

            this.iter = Primitives.parseInt(cl.getOptionValue("iter"), 40);
            if (this.iter <= 0) {
                throw new UDFArgumentException("Argument `int iter` must be greater than 0: "
                        + this.iter);
            }

        } else {
            this.l1 = 0.01d;
            this.l2 = 0.01d;
            this.iter = 40;
        }
        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        int i = PrimitiveObjectInspectorUtils.getInt(args[0], itemIOI);
        Map Ri = this.itemIRatesOI.getMap(args[1]);
        Map topKRatesOfI = this.topKRatesOfIOI.getMap(args[2]);
        int j = PrimitiveObjectInspectorUtils.getInt(args[3], itemJOI);
        Map Rj = this.itemJRatesOI.getMap(args[4]);
        train(i, Ri, topKRatesOfI, j, Rj);
    }

    @Override
    public void close() throws HiveException {
        int numItem = Math.max(this.W.numRows(), this.W.numColumns());
        for (int i = 0; i < numItem; i++) {
            for (int j = 0; j < numItem; j++) {
                if (this.W.unsafeGet(i, j, 0.) != 0.) {
                    Object[] res = new Object[3];
                    res[0] = new IntWritable(i);
                    res[1] = new IntWritable(j);
                    res[2] = new DoubleWritable(this.W.get(i, j));
                    forward(res);
                }
            }
        }
    }

    protected double predict(Object u, int i, Map<?, ?> topKRatesOfI, int excludeIndex) {
        if (!topKRatesOfI.containsKey(u)) {
            return 0.;
        }
        double pred = 0.d;
        for (Map.Entry<?, ?> rating : this.topKRatesOfIValueOI.getMap(topKRatesOfI.get(u))
                                                              .entrySet()) {
            int k = PrimitiveObjectInspectorUtils.getInt(rating.getKey(),
                this.topKRatesOfIValueKeyOI);
            if (k == excludeIndex) {
                continue;
            }
            double rate = PrimitiveObjectInspectorUtils.getDouble(rating.getValue(),
                this.topKRatesOfIValueValueOI);
            pred += rate * this.W.unsafeGet(i, k, 0.d);
        }
        return pred;
    }

    protected double predict(Object u, int i, Map<?, ?> topKRatesOfI) {
        if (!topKRatesOfI.containsKey(u)) {
            return 0.;
        }

        double pred = 0.d;
        for (Map.Entry<?, ?> rating : this.topKRatesOfIValueOI.getMap(topKRatesOfI.get(u))
                                                              .entrySet()) {
            int k = PrimitiveObjectInspectorUtils.getInt(rating.getKey(),
                this.topKRatesOfIValueKeyOI);
            double rate = PrimitiveObjectInspectorUtils.getDouble(rating.getValue(),
                this.topKRatesOfIValueValueOI);

            pred += rate * this.W.unsafeGet(i, k, 0.d);
        }
        return pred;
    }

    private void train(int i, Map<?, ?> Ri, Map<?, ?> topKRatesOfI, int j, Map<?, ?> Rj) {
        int N = Rj.size();
        double gradSum = 0.d;
        double rateSum = 0.d;
        double errs = 0.d;
        for (Map.Entry<?, ?> userRate : Rj.entrySet()) {
            Object u = userRate.getKey();
            double ruj = PrimitiveObjectInspectorUtils.getDouble(userRate.getValue(),
                this.itemJRateValueOI);
            double rui = 0.d;
            if (Ri.containsKey(u)) {
                rui = PrimitiveObjectInspectorUtils.getDouble(Ri.get(u), this.itemIRateValueOI);
            }

            double eui = rui - predict(u, i, topKRatesOfI, j);
            gradSum += ruj * eui;
            rateSum += ruj * ruj;
            errs += eui * eui;
        }

        gradSum /= N;
        rateSum /= N;
        errs /= N;

        double update = 0.d;
        if (this.l1 < Math.abs(gradSum)) {
            if (gradSum > 0.) {
                update = (gradSum - this.l1) / (rateSum + this.l2);
            } else {
                update = (gradSum + this.l1) / (rateSum + this.l2);
            }
            if (update < 0.) { // non-negativity constraints
                update = 0.;
            }
        }

        this.loss += errs;
        this.W.unsafeSet(i, j, update);
    }

    public void resetLoss() {
        this.loss = 0.d;
    }

    public double getLoss() {
        return this.loss;
    }
}
