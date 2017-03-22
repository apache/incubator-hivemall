/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package hivemall.lda;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

public final class OnlineLDAModel {
	
	//
	boolean printLambda = false;
	boolean printGamma  = false;
	boolean printPhi    = false;

	
	// Free Parameters
	private int K_;	
	
	// 
	private int D_;	
	private int tmpTotalD_ = 11102;
	private float accTotalD = 0;
	private int accTotalWords = 0;
	
	//
	ArrayList<HashMap<String, float[]>> phi_;
	float[][]   gamma_;
	HashMap<String, float[]> lambda_; 
	
	// Random Variables
	GammaDistribution gd;
	
	// CONSTANTS
	private double SHAPE = 100d;
	private double SCALE = 1d / SHAPE;
	
	private double DELTA = 1E-5;
	
	private double tau0_ = 1020;
	private double kappa_= 0.7;
	private double rhot;
	private static float alpha_ = 1/2f;
	private float eta_= 1/ 20f;
	
	private int dummySize = 100;

	private float[][] dummyLambdas;
	
//	private int batchSize_;
	
	private ArrayList<HashMap<String, Float>> miniBatchMap; 
	
	
//	private String[] Symbols = {"\\", "/", ">" ,"<" ,"-" ,"," ,"." ,"(" ,")" ,":" ,";" ,"'" ,"[" ,"]","!" ,"*" ,"#" ,"+","%" ,"@","&","?","$","0","1","2","3","4","5","6","7","8","9","\t","_","{","}","=","|"};
	
	// Constructor
	public OnlineLDAModel(int K, double alpha, double eta, int totalD, double tau0, double kappa, int batchSize){
		// Initialize Free Params
		K_ = K;
		alpha_ = (float)alpha;
		eta_ = (float)eta;
		tmpTotalD_ = totalD;
		tau0_ = tau0;
		kappa_ = kappa;
//		batchSize_= batchSize;
		
		// Initialize Internal Params
		setRandomParams();
		setParams();
		setDummyLambda();
	}

	private void setDummyLambda() {
		dummyLambdas = new float[dummySize][];
		double[] tmpDArray;
		for(int b=0; b<dummySize; b++){
			float[] tmpDummyLambda = new float[K_];
			tmpDArray = gd.sample(K_); 
			for(int k=0; k<K_; k++){
				tmpDummyLambda[k] = (float)tmpDArray[k];
			}
			dummyLambdas[b] = tmpDummyLambda;
		}
	}

	private void setParams() {
		lambda_ = new HashMap<String, float[]>();
	}


	private void setRandomParams() {
		gd = new GammaDistribution(SHAPE, SCALE);
		gd.reseedRandomGenerator(1001);
		for(int i=0; i<1000; i++){
			double[] tmpD = gd.sample(20);
			float[] tmp = new float[tmpD.length];
			for(int d=0; d<tmpD.length; d++){
				tmp[d] = (float)tmpD[d];
			}
			tmp[0] = tmp[1];
		}
	}


	private void getMiniBatchParams(String[][] miniBatch) {
		D_ = miniBatch.length;
		
		for(int d=0; d<D_; d++){
			 accTotalWords = miniBatch[d].length;
		}
	}

	private void updateSizeOfParameterPerMiniBatch() {
		// phi_
		phi_ = new ArrayList<HashMap<String, float[]>>();
		gamma_ = new float[D_][];
		// gamma and phi
		for(int d=0; d<D_; d++){
			// Gamma
			float [] gammad = getRandomGammaArray();
			gamma_[d] = gammad;

			// phi_ not needed to be initialized
			HashMap<String, float[]> phid = new HashMap<String, float[]>();
			for(String label:miniBatchMap.get(d).keySet()){
				float[] tmpFArray = new float[K_];
				phid.put(label, tmpFArray);
			}
			phi_.add(phid);
		}
		
		// lambda
		for(int d=0; d<D_; d++){
			for(String label:miniBatchMap.get(d).keySet()){
				if(!lambda_.containsKey(label)){
					int tmpLambdaIdx = lambda_.size() % dummySize;
					float[] lambdaNW = getRandomGammaArray();
					for(int k=0; k<K_; k++){
						lambdaNW[k] *= dummyLambdas[tmpLambdaIdx][k];
					}
					lambda_.put(label, lambdaNW);
				}
			}
		}
	}

	private float[] getRandomGammaArray() {
		double[] dret = new double[K_];
		float[] ret = new float[K_];

		dret = gd.sample(K_);

		for(int k=0; k<ret.length; k++){
			ret[k] = (float)dret[k];
		}

		return ret;
	}

	private void do_m_step() {
		// calculate lambdaBar
		HashMap<String, float[]> lambdaBar = new HashMap<String, float[]>();

		float multiplier = ((float)tmpTotalD_ / (float)D_);
		for(int d=0; d<D_; d++){
			for(String label:miniBatchMap.get(d).keySet()){
				if(!lambdaBar.containsKey(label)){
					float[] tmp = new float[K_];
					Arrays.fill(tmp, eta_);
					for(int k=0; k<K_; k++){
						tmp[k] += multiplier * phi_.get(d).get(label)[k];
					}
					lambdaBar.put(label, tmp);
				}else{
					float[] tmp = lambdaBar.get(label);
					for(int k=0; k<K_; k++){
						tmp[k] += multiplier * phi_.get(d).get(label)[k];
					}
					lambdaBar.put(label, tmp);
				}
			}
		}
		
		// update
		float oneMinuxRhot = (float)(1 - rhot);
		for(String label:lambda_.keySet()){
			float[] tmp = lambda_.get(label);
			if(!lambdaBar.containsKey(label)){
				for(int k=0; k<K_; k++){
					tmp[k] = oneMinuxRhot * tmp[k] + (float)rhot * eta_;
				}
			}else{
				float[] tmpLambda = lambdaBar.get(label);
				for(int k=0; k<K_; k++){
					tmp[k] = (float)(oneMinuxRhot * tmp[k] + rhot * tmpLambda[k]);
				}
			}
			lambda_.put(label, tmp);
		}
	}
	
	private void do_e_step_phi(){
		// Calc Theta
		float[][] eLogTheta = new float[D_][K_];
		for(int d=0; d<D_; d++){
			// calc D sum
			float tmpSum = 0;
			float dSum = 0;
			for(int k=0; k<K_; k++){
				dSum += gamma_[d][k];
			}
			float gamma_dSum = (float)Gamma.digamma(dSum);

			for(int k=0; k<K_; k++){
				eLogTheta[d][k] = (float)(Gamma.digamma(gamma_[d][k]) - gamma_dSum);
				tmpSum += eLogTheta[d][k];
			}
		}
		
		// Calc Beta
		HashMap<String, float[]> elogBeta = new HashMap<String, float[]>();
		for(int k=0; k<K_; k++){
			// calc K sum
			float kSum = 0;
			for(String label:lambda_.keySet()){
				kSum += lambda_.get(label)[k];
			}
			float gamma_kSum = (float)Gamma.digamma(kSum);

			for(int d=0; d<D_; d++){
				for(String label: miniBatchMap.get(d).keySet()){
					if(elogBeta.containsKey(label)){
						float[] tmpArray = elogBeta.get(label);
						tmpArray[k] = (float)Gamma.digamma(lambda_.get(label)[k]) - gamma_kSum;
						elogBeta.put(label, tmpArray);
					}else{
						float[] tmpArray = new float[K_];
						Arrays.fill(tmpArray, 0f);
						tmpArray[k] = (float)Gamma.digamma(lambda_.get(label)[k]) - gamma_kSum;
						elogBeta.put(label, tmpArray);
					}
				}
			}
		}
		
		for(int d=0; d<D_; d++){
			for(String label:miniBatchMap.get(d).keySet()){
			float normalizer = 0;
				for(int k=0; k<K_; k++){
					phi_.get(d).get(label)[k] = (float) Math.exp(eLogTheta[d][k] + elogBeta.get(label)[k]) + 1E-20f;
					normalizer += phi_.get(d).get(label)[k];
				}
				// normalize 
				for(int k=0; k<K_; k++){
					phi_.get(d).get(label)[k] /= normalizer;
				}
			}
		}
	}

	
	public void showTopicWords() {
		System.out.println("SHOW TOPIC WORDS:");
		System.out.println("WORD SIZE:" + lambda_.size());
		for(int k=0; k<K_; k++){
			
			float lambdaSum = 0;
			for(String label:lambda_.keySet()){
				lambdaSum += lambda_.get(label)[k];
			}
			
			System.out.print("Topic:" + k);

			System.out.println("===================================");
			ArrayList<String> sortedWords = getSortedLambda(k);
			System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
			int topN = Math.min(50, lambda_.keySet().size());
			for(int tt=0; tt<topN; tt++){
				String label = sortedWords.get(tt);
				System.out.println("No." + tt + "\t" + label + "[" + label.length() + "]" + ":\t" + lambda_.get(label)[k] / lambdaSum);
			}
			System.out.println("==========================================");
		}
	}

	private ArrayList<String> getSortedLambda(int k) {
		ArrayList<String> ret = new ArrayList<String>();
		ArrayList<LabelValueTuple> compareList = new ArrayList<LabelValueTuple>();
		
		for(String label:lambda_.keySet()){
			float tmpValue = lambda_.get(label)[k];
			compareList.add(new LabelValueTuple(label, tmpValue));
		}

		Collections.sort(compareList, new LabelValueTupleComparator());
		
		for(int w=0,W=compareList.size(); w<W; w++){
			String label = compareList.get(w).getLabel();
			ret.add(label);
		}
		return ret;
	}	
	
	public float getPerplexity(){
		float ret = 0;
		float bound = calcBoundPerMiniBatch();
		
		ret = (float) Math.exp((-1) * ((float)bound / (float)accTotalWords));
		
		return ret;
	}
	
	public int[] getMaxGammaGroup(int D){
		int[] ret = new int[D];
		
		for(int d=0; d<D; d++){
			int tmpK = -1;
			float tmpGammaK = -1;
			for(int k=0; k<K_; k++){
				if(tmpGammaK < gamma_[d][k]){
					tmpK = k;
					tmpGammaK = gamma_[d][k];
				}
			}
			ret[d] = tmpK;
		}
		return ret;
	}
	
	private float calcBoundPerMiniBatch(){
		float ret = 0;
		
		float tmpSum1 = 0;
		float tmpSum2 = 0;
		float tmpSum3 = 0;
		float tmpSum4 = 0;

		float tmpSum3_1 = 0;
		float tmpSum3_2 = 0;
		
		float tmpSum3_2_1 = 0;
		float tmpSum3_2_2 = 0;
		float tmpSum3_2_3 = 0;
		
		
		float tmpSum4_1 = 0;
		float tmpSum4_2 = 0;
		float tmpSum4_3 = 0;
		float tmpSum4_4 = 0;
		
		// Prepare
		float[] gammaSum = new float[D_];
		Arrays.fill(gammaSum, 0f);
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				gammaSum[d] += gamma_[d][k];
			}
		}
		float[] lambdaSum = new float[K_];
		Arrays.fill(lambdaSum, 0f);
		for(int k=0; k<K_; k++){
			for(String label:lambda_.keySet()){
				lambdaSum[k] = lambda_.get(label)[k];
			}
		}

		// Calculate
		for(int d=0; d<D_; d++){

			// FIRST LINE **
			for(String label:miniBatchMap.get(d).keySet()){
				float ndw = miniBatchMap.get(d).get(label);
				float  EqlogTheta_dk = 0;
				float  EqlogBeta_kw = 0;
				tmpSum1 = 0;

				for(int k=0; k<K_; k++){
					try{
						EqlogTheta_dk = (float)(Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]));
						EqlogBeta_kw  = (float)(Gamma.digamma(lambda_.get(d)[k] - Gamma.digamma(lambdaSum[k])));
					}catch(Exception e){
//						System.out.println("gamma[d][k]:" + gamma_[d][k]);
//						System.out.println(EqlogTheta_dk);
//						System.out.println(EqlogBeta_kw);
					}

					tmpSum1 += phi_.get(d).get(label)[k] * (EqlogTheta_dk + EqlogBeta_kw - Math.log(phi_.get(d).get(label)[k]));
				}
				ret += ndw * tmpSum1;	// 1-1
			}
			// ** FIRST LINE
			
			// SECOND LINE **
			ret -= (Gamma.logGamma(gammaSum[d]));	// 2-1
			tmpSum2 = 0;
			for(int k=0; k<K_; k++){
				tmpSum2 += (alpha_ - gamma_[d][k]) 
						* (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]))
						+(Gamma.logGamma(gamma_[d][k]));
				tmpSum2 /= accTotalD;
			}
			ret += tmpSum2;			// 2-2
			// ** SECOND LINE

			// THIRD LINE **
			tmpSum3 = 0;
			for(int k=0; k<K_; k++){
				tmpSum3_1 = 0;
				tmpSum3_2 = 0;
				
				tmpSum3_1 = (-1f) * (float)Gamma.logGamma(lambdaSum[k]);
				for(String label:lambda_.keySet()){
					tmpSum3_2_1 = (eta_ - lambda_.get(label)[k]);
					tmpSum3_2_2 = (float)(Gamma.digamma(lambda_.get(label)[k]) - Gamma.digamma(lambdaSum[k]));
					tmpSum3_2_3 = (float)(Gamma.logGamma(lambda_.get(label)[k]));
					tmpSum3_2 += (tmpSum3_2_1 * tmpSum3_2_2 * tmpSum3_2_3);
				}
				
				tmpSum3 += (tmpSum3_1 - tmpSum3_2);
			}
			ret += (tmpSum3 / D_);		//3-1
			// ** THIRD LINE
			
			// FOURTH LINE **
			float W = lambda_.size();
			tmpSum4_1 = (float)(Gamma.logGamma(K_ * alpha_));
			tmpSum4_2 = (float)(K_ * (Gamma.logGamma(alpha_)));
			tmpSum4_3 = (float)(Gamma.logGamma(W * eta_));
			tmpSum4_4 = (float)((-1) * W * (Gamma.logGamma(eta_))); 
			tmpSum4 =  tmpSum4_1 - tmpSum4_2 + ((tmpSum4_3 - tmpSum4_4) / accTotalD); 
			ret += tmpSum4;
			// ** FOURTH LINE
		}
		return ret;
	}

	public void train(String[][] miniBatch, int time) {
		
		D_ = miniBatch.length;
		
		rhot = Math.pow(tau0_ + time, -kappa_);
	
		if(printLambda){
			System.out.println("Lambda:");
			for(String key: lambda_.keySet()){
				System.out.println(Arrays.toString(lambda_.get(key)));
			}
		}
		
		// get the number of words(Nd) for each documents
		getMiniBatchParams(miniBatch);
		accTotalD += D_;
		
		makeMiniBatchMap(miniBatch);

		updateSizeOfParameterPerMiniBatch();
		
		// E STEP
		float[][] lastGamma = new float[D_][K_];
		do{
			lastGamma = copyMatrix(gamma_);
			do_e_step_phi();
			do_e_step_gamma();
		}while(!checkGammaDiff(lastGamma, gamma_));
		
		// M Step
		do_m_step();
		
		if(printGamma){
			System.out.println("Gamma:");
			for(int d=0; d<D_; d++){
				for(int w=0; w<gamma_[d].length; w++){
					System.out.print(gamma_[d][w] + ",");
				}
				System.out.println("");
			}
		}

		if(printPhi){
			System.out.println("phi");
			for(int d=0; d<D_; d++){
				for(String label:miniBatchMap.get(d).keySet()){
					System.out.println(Arrays.toString(phi_.get(d).get(label)));
				}
			}
		}
	}

	private boolean checkGammaDiff(float[][] lastGamma, float[][] nextGamma) {
		double diff = 0;
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				diff += (float)Math.abs(lastGamma[d][k] - nextGamma[d][k]);
			}
		}
		if(diff < DELTA * D_ * K_){
			return true;
		}else{
			return false;
		}
	}

	private void do_e_step_gamma() {
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				float gamma_tk = alpha_;
				for(String label:miniBatchMap.get(d).keySet()){
					gamma_tk += phi_.get(d).get(label)[k] * miniBatchMap.get(d).get(label);
				}
				gamma_[d][k] = gamma_tk;
			}
		}
	}

	private float[][] copyMatrix(float[][] gamma_2) {
		float[][] ret = new float[gamma_2.length][];
		for(int d=0; d<gamma_2.length; d++){
			float[] tmpd = new float[gamma_2[d].length];
			for(int k=0; k<gamma_2[d].length; k++){
				tmpd[k] = gamma_2[d][k];
			}
			ret[d] = tmpd;
		}
		return ret;
	}

	private void makeMiniBatchMap(String[][] miniBatch) {
		miniBatchMap = new ArrayList<HashMap<String, Float>>();
		for(int d=0; d<D_; d++){
			HashMap<String, Float> mapd = new HashMap<String, Float>();
			for(int w=0; w<miniBatch[d].length; w++){
				String tmpString = miniBatch[d][w];
				String[] label_value = tmpString.split(":");
				if(label_value.length == 1){
					continue;
				}
				String label = label_value[0];
				float  value = Float.parseFloat(label_value[1]);
				mapd.put(label, value);
			}
			miniBatchMap.add(mapd);
		}
	}
}