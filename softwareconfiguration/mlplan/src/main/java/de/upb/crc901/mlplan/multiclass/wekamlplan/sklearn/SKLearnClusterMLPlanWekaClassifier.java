package de.upb.crc901.mlplan.multiclass.wekamlplan.sklearn;

import java.io.IOException;

import de.upb.crc901.mlplan.core.AbstractMLPlanBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;

public class SKLearnClusterMLPlanWekaClassifier extends MLPlanWekaClassifier {

	public SKLearnClusterMLPlanWekaClassifier(final AbstractMLPlanBuilder builder) throws IOException {
		super(builder);
	}

	public SKLearnClusterMLPlanWekaClassifier() throws IOException {
		super(AbstractMLPlanBuilder.forSKLearnCluster());
	}

}
