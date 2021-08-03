package edu.metu.ceng790.project;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowth.FreqItemset;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.sql.SparkSession;

public class AssocRules {

    public static void main(String[] args) {
        SparkContext sc = SparkSession.builder().appName("PrefixSpan").config("spark.master", "local[*]").getOrCreate().sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        JavaRDD<String> data = jsc.textFile("data\\msnbc_fpgrowth.txt");

        JavaRDD<List<String>> transactions = data.map(line -> Arrays.asList(line.split(" ")));

        FPGrowth fpg = new FPGrowth()
                .setMinSupport(0.2)
                .setNumPartitions(10);
        FPGrowthModel<String> model = fpg.run(transactions);

        JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = model.freqItemsets().toJavaRDD();

        /*
        JavaRDD<FPGrowth.FreqItemset<String>> freqItemsets = jsc.parallelize(Arrays.asList(
                new FreqItemset<String>(new String[] {"a"}, 15L),
                new FreqItemset<String>(new String[] {"b"}, 35L),
                new FreqItemset<String>(new String[] {"a", "b"}, 12L)
        ));
        */

        AssociationRules arules = new AssociationRules().setMinConfidence(0.8);
        JavaRDD<AssociationRules.Rule<String>> results = arules.run(freqItemsets);

        for (AssociationRules.Rule<String> rule : results.collect()) {
            System.out.println(
                    rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence());
        }
    }
}
