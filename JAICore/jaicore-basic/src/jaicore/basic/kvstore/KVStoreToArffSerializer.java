package jaicore.basic.kvstore;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class KVStoreToArffSerializer {

	public static String serialize(final KVStoreCollection collection, final String classFieldName) {
		// adapted from WekaUtil, fromJAICoreInstances(final WekaCompatibleInstancesImpl instances) method

		/* create basic attribute entries */
		final ArrayList<Attribute> attributes = new ArrayList<>();
		final int numAttributes = collection.size();

		// assuming every kvstore has the same column names:
		final List<String> columns = new LinkedList<>(collection.getFirst().keySet());

		//error: if(!columns.contains(classFieldName)){return null;}
		columns.remove(classFieldName);
		columns.add(classFieldName);    // make sure the label is the last element
		final List<Boolean> nominal = new LinkedList<>();
		final HashMap<String, Map<Object, Double>> maps = new HashMap<>();
		for (final String attrName : columns) {


			/*
			 * encode values
			 */
			final Map<Object, Double> labelMap = new HashMap<>();
			int c = 0;
			boolean isNominal = false;
			for (final Object o : collection.stream().map(x -> x.get(attrName)).collect(Collectors.toList())) {
				if (labelMap.containsKey(o)) {
					continue;
				}
				labelMap.put(o, (double) (c++));
				if (!Double.class.isInstance(o)) {
					isNominal = true;
				}
			}

			/* if the feature is */
			if (isNominal) {
				attributes.add(new Attribute(attrName, new LinkedList<>(labelMap.keySet().stream().map(x -> (String) x).collect(Collectors.toList()))));
			} else {
				attributes.add(new Attribute(attrName));
			}
			nominal.add(isNominal);
			maps.put(attrName, labelMap);
		}

		/* create instances object and insert the data points */
		final Instances wekaInstances = new Instances("KVStore - exported dataset", attributes, collection.size());
		wekaInstances.setClassIndex(numAttributes);
		for (final KVStore instance : collection) {
			final Instance wekaInstance = new DenseInstance(numAttributes + 1);
			wekaInstance.setDataset(wekaInstances);
			int att = 0;
			for (final String attrName : columns) {
				if (attrName.equals(classFieldName)) {
					wekaInstance.setClassValue(maps.get(attrName).get(instance.get(classFieldName)));
					continue;
				}
				if (nominal.get(att)) {
					wekaInstance.setValue(att++, maps.get(attrName).get(instance.get(attrName)));
				} else {
					wekaInstance.setValue(att++, instance.getAsDouble(attrName));
				}
				wekaInstances.add(wekaInstance);
			}

		}
		return wekaInstances.toString();
	}
}
