package ai.libs.jaicore.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import weka.core.Instances;

public class ClusterUtil {
	public static int[][] calculateContingencyMatrix(List<Double> actual, List<Double> expected){
		//assumption: actual and expected have the same data sorted identically, cluster indices range from 0 to clustercount
		//count clusters
		Map<Integer,Integer> actualLabelIndexMap = new HashMap<>();
		Map<Integer,Integer> expectedLabelIndexMap = new HashMap<>();

		int currentTableIndex = 0;
		for (int i = 0; i < actual.size(); i++) {
			int actualLabelClusterIndex = (int)actual.get(i).doubleValue();
			if(!actualLabelIndexMap.containsKey(actualLabelClusterIndex)){
				actualLabelIndexMap.put(actualLabelClusterIndex, currentTableIndex);
				currentTableIndex++;
			}
		}
		currentTableIndex = 0;
		for (int i = 0; i < expected.size(); i++) {
			int expectedLabelClusterIndex = (int)expected.get(i).doubleValue();
			if(!expectedLabelIndexMap.containsKey(expectedLabelClusterIndex)){
				expectedLabelIndexMap.put(expectedLabelClusterIndex,currentTableIndex);
				currentTableIndex++;
			}
		}
		//create table
		int[][] contingency = new int[actualLabelIndexMap.size()][expectedLabelIndexMap.size()];
		for (int i = 0; i < actual.size(); i++) {
			int actualLabelClusterIndex = (int)actual.get(i).doubleValue();
			int actualLabelTableIndex = actualLabelIndexMap.get(actualLabelClusterIndex);
			int expectedLabelClusterIndex = (int)expected.get(i).doubleValue();
			int expectedLabelTableIndex = expectedLabelIndexMap.get(expectedLabelClusterIndex);
			contingency[actualLabelTableIndex][expectedLabelTableIndex] = contingency[actualLabelTableIndex][expectedLabelTableIndex] + 1;
		}
		return contingency;
	}
	public static List<List<double[]>> separateClusters(Instances labeledData){

		List<List<double[]>> clusters = new LinkedList<>();
		Map<Double, Integer> label2index = new HashMap<>();
		for (int i = 0; i < labeledData.size(); i++) {
			if(!label2index.containsKey(labeledData.get(i).value(labeledData.classIndex()))){
				clusters.add(new LinkedList<>());
				label2index.put(labeledData.get(i).value(labeledData.classIndex()), clusters.size()-1);
			}
			int clusterIndex = label2index.get(labeledData.get(i).value(labeledData.classIndex()));
			clusters.get(clusterIndex).add(labeledData.get(i).toDoubleArray());
		}

		return clusters;
	}

	public static double[] calculateCentroid(List<double[]> cluster){
		if(cluster.size() == 0){
			throw new IllegalArgumentException("the cluster needs at least one element");
		}
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
	public static List<double[]> calculateCentroids(List<List<double[]>> clusters){
		List<double[]> centroids = new ArrayList<>();
		for (List<double[]> cluster : clusters
		) {
			centroids.add(calculateCentroid(cluster));
		}
		return centroids;
	}
}
