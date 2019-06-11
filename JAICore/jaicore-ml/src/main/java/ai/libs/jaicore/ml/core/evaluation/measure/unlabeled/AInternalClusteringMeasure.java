package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import jaicore.basic.aggregate.IAggregateFunction;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import weka.core.Instances;

public abstract class AInternalClusteringMeasure implements IInternalClusteringMeasure {
	public List<List<double[]>> separateClusters(Instances labels, Instances data){
		if(labels.size() != data.size()){
			throw new IllegalArgumentException("labels and data have different length");
		}
		List<List<double[]>> clusters = new LinkedList<>();
		Map<Double, Integer> label2index = new HashMap<>();
		for (int i = 0; i < data.size(); i++) {
			if(!label2index.containsKey(labels.get(i).value(0))){
				clusters.add(new LinkedList<>());
				label2index.put(labels.get(i).value(0), clusters.size()-1);
			}
			int clusterIndex = label2index.get(labels.get(0).value(0));
			clusters.get(clusterIndex).add(data.get(i).toDoubleArray());
		}
		return clusters;
	}

	public double[] calculateCentroid(List<double[]> cluster){
		// calculate centroid
		double[] centroid = new double[cluster.get(0).length];
		for (int i = 0; i < cluster.size(); i++) {
			for (int j = 0; j < cluster.get(i).length; j++) {
				centroid[j] += cluster.get(i)[j];
			}
		}
		for (int i = 0; i < centroid.length; i++) {
			centroid[i] /= cluster.size();
		}
		return centroid;
	}
// euclid's distance
	public double distance(double[] vecA, double[] vecB){
		if(vecA.length != vecB.length){
			throw new IllegalArgumentException("the vectors have different dimensions");
		}
		double sqsum = 0;
		for (int i = 0; i < vecA.length; i++) {
			sqsum += Math.pow(vecA[i]-vecB[i],2);
		}
		return Math.sqrt(sqsum);
	}

	public abstract Double calculateMeasure(List<List<double[]>> clusters);

	@Override
	public Double calculateMeasure(final Instances labels, final Instances data) {
		return calculateMeasure(separateClusters(labels,data));
	}

	@Override
	public List<Double> calculateMeasure(final List<Instances> actual, final List<Instances> expected) {
		return null;
	}

	@Override
	public Double calculateMeasure(final List<Instances> actual, final List<Instances> expected, final IAggregateFunction<Double> aggregateFunction) {
		return null;
	}

	@Override
	public Double calculateAvgMeasure(final List<Instances> actual, final List<Instances> expected) {
		return null;
	}
}