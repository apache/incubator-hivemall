package hivemall.lda;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.json.CDL;

import hivemall.UDTFWithOptions;
import hivemall.utils.HivemallUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

public class OnlineLDAUDTF extends UDTFWithOptions {

    private ListObjectInspector _XOI;

    // Options
    protected int topic;
    protected float alpha;
    protected float eta;
    protected int numDoc;
    protected double tau0;
    protected double kappa;
    protected double delta;

    private int topics; // The number of topics
    private double alpha; // The hyperparameter for theta
    private double eta; // The hyperparameter for beta
    private int _totalD; // The total number of Document
    private double _tau0; // The parameter to control learning speed of Lambda
    private double _kappa; // The parameter to control learning speed of Lambda
    private int _batchSize; // The mini-batch size

    private int _iterations;// The number of Iterations
    private double _delta; // The number for Convergence check


    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k", "topic", true, "The number of topics");
        opts.addOption("alpha", true, "The hyperparameter for theta [Default: 1/K]");
        opts.addOption("eta", true, "The hyperparameter for beta [Default: 1/K]");
        opts.addOption("d", "num_doc", true, "The total number of documents [Default: 10000]");
        opts.addOption("tau", "tau0", true,
            "The parameter which downweights early iterations [Default: 64]");
        opts.addOption("kappa", true, "Exponential decay rate (i.e., learning rate) [Default: 0.7]");
        opts.addOption("iter", "iterations", true, "The number of Iterations [default: 1]");
        opts.addOption("delta", false, "The number for Convergence check [default: 1E-5]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        int topics = 10;
        double alpha = 1. / topics;
        double eta = 1. / topics;
        int totalD = 10000;
        double tau0 = 64;
        double kappa = 0.7;
        int iterations = 1;
        double delta = 1E-5;

        CommandLine cl = null;
        if (argOIs.length >= 2) {
            String rawArgs = HiveUtils.getConstString(argOIs[1]);
            cl = parseOptions(rawArgs);
            topics = Primitives.parseInt(cl.getOptionValue("topics"), topics);
            alpha = Primitives.parseDouble(cl.getOptionValue("alpha"), alpha);
            eta = Primitives.parseDouble(cl.getOptionValue("eta"), eta);
            totalD = Primitives.parseInt(cl.getOptionValue("totalD"), totalD);
            tau0 = Primitives.parseDouble(cl.getOptionValue("tau0"), tau0);
            kappa = Primitives.parseDouble(cl.getOptionValue("kappa"), kappa);
            iterations = Primitives.parseInt(cl.getOptionValue("iterations"), iterations);
            delta = Primitives.parseDouble(cl.getOptionValue("delta"), delta);
        }

        this._topics = topics;
        this._alpha = alpha;
        this._eta = eta;
        this._totalD = totalD;
        this._kappa = kappa;
        this._iterations = iterations;
        this._delta = delta;

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        // TODO Auto-generated method stub
        //		if(argOIs.length!=1 && argOIs.length!=2){
        //			throw new UDFArgumentException(getClass().getSimpleName()
        //					+ "takes 1 or 2 arguments: "	// TODO ask Yui san
        //					);
        //		}
        return null;
    }

    @Override
    public void close() throws HiveException {
        // TODO Auto-generated method stub

    }



    @Override
    public void process(Object[] arg0) throws HiveException {
        // TODO Auto-generated method stub

    }


}
