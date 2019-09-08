package ai.libs.jaicore.ml.weka.dataset.splitter;

import ai.libs.jaicore.ml.WekaUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instances;

/**
 * Generates a purely random split of the dataset depending on the seed and on the portions provided.
 *
 * @author mwever
 */
public class ArbitrarySplitter implements IDatasetSplitter {

	@Override
	public List<Instances> split(final Instances data, final long seed, final double... portions) {
		// if we copy, scikitlearnwrapper will always create a new file
		if (portions.length == 1 && portions[0] == 0) {
			final ArrayList<Instances> ret = new ArrayList<>();
			ret.add(null);
			ret.add(data);
		}
		return WekaUtil.realizeSplit(data, WekaUtil.getArbitrarySplit(data, new Random(seed), portions));
	}

}
