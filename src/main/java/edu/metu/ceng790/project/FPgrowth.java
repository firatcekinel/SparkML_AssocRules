package edu.metu.ceng790.project;

import java.util.*;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;
import org.apache.spark.sql.SparkSession;

public class FPgrowth {


    public static void main(String[] args) {
        SparkContext sc = SparkSession.builder().appName("FPGrowth").config("spark.master", "local[*]").getOrCreate().sparkContext();
        sc.setLogLevel("ERROR");
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        Map<Integer, String> pageMap = new HashMap<>();
        pageMap.put(1, "frontpage");
        pageMap.put(2, "news");
        pageMap.put(3, "tech");
        pageMap.put(4, "local");
        pageMap.put(5, "opinion");
        pageMap.put(6, "on-air");
        pageMap.put(7, "misc");
        pageMap.put(8, "weather");
        pageMap.put(9, "msn-news");
        pageMap.put(10, "health");
        pageMap.put(11, "living");
        pageMap.put(12, "business");
        pageMap.put(13, "msn-sports");
        pageMap.put(14, "sports");
        pageMap.put(15, "summary");
        pageMap.put(16, "bbs");
        pageMap.put(17, "travel");

        JavaRDD<String> data = jsc.textFile("data\\msnbc_fpgrowth_label.txt");

        JavaRDD<List<String>> transactions = data.map(line -> Arrays.asList(line.split(" ")));

        FPGrowth fpg = new FPGrowth()
                .setMinSupport(0.0001)
                .setNumPartitions(10);
        FPGrowthModel<String> model = fpg.run(transactions);

        System.out.println("DISPLAY FREQUENT PAGES!!!");
        Map<String, Double> freqItemsetMap = new HashMap<>();
        // display frequent itemsets
        for (FPGrowth.FreqItemset<String> itemset: model.freqItemsets().toJavaRDD().collect()) {
            //System.out.println("[" + itemset.javaItems() + "], " + itemset.freq());
            if (itemset.javaItems().size() == 1)
                freqItemsetMap.put(itemset.javaItems().get(0), itemset.freq()*1.0/transactions.count());
        }

        // rule generation -> AssociationRules algoritmasında consequent tek item oluyor
        System.out.println("********************** GENERATING RULES!!! *************************");
        double minConfidence = 0.6;

        List<AssociationRules.Rule<String>> ruleList = new ArrayList<>(model.generateAssociationRules(minConfidence).toJavaRDD().collect());
        ruleList.sort(Comparator.comparing(AssociationRules.Rule<String>::confidence).reversed());

        System.out.println("RULE COUNT = " + ruleList.size());
        // display top-k rules
        System.out.println("RULE, CONFIDENCE, INTEREST, LIFT");
        int topK_rules = 10;
        for (AssociationRules.Rule<String> rule : ruleList) {
            if (topK_rules-- == 0) break;
            // Interest(I -> j) = conf(I -> j) - Pr[ j]
            double interest = Math.abs(rule.confidence() - freqItemsetMap.get(rule.javaConsequent().get(0)));
            // lift(A->B) = P(A,B) / (P(A)*P(B))
            double lift = rule.confidence() / freqItemsetMap.get(rule.javaConsequent().get(0));
            // ANTECEDENT -> CONSEQUENT, CONFIDENCE, INTEREST
            System.out.println(rule.javaAntecedent() + " => " + rule.javaConsequent() + ", " + rule.confidence()
                    + ", " + interest + ", " + lift);

        }
    }
}
