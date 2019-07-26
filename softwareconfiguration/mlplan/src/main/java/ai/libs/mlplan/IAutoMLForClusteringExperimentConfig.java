package ai.libs.mlplan;


import ai.libs.jaicore.experiments.IDatabaseConfig;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import java.io.File;
import org.aeonbits.owner.Config.Sources;


@Sources({"classpath:automlForClusteringExperiment.properties"})
public interface IAutoMLForClusteringExperimentConfig extends IExperimentSetConfig, IDatabaseConfig {


	@Key("datasetDir")
	public File datasetDirectory();


}
