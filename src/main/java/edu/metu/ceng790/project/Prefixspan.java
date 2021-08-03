package edu.metu.ceng790.project;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.fpm.AssociationRules;
import org.apache.spark.mllib.fpm.PrefixSpan;
import org.apache.spark.mllib.fpm.PrefixSpanModel;
import org.apache.spark.sql.SparkSession;
import scala.collection.Seq;

public class Prefixspan {

    public static void main(String[] args) throws FileNotFoundException {
        SparkContext sc = SparkSession.builder().appName("PrefixSpan").config("spark.master", "local[*]").getOrCreate().sparkContext();
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

        //JavaRDD<String> data = jsc.textFile("data\\msnbc_fpgrowth.txt");
        //JavaRDD<List<String>> transactions = data.map(line -> Arrays.asList(line.split(" ")));

        BufferedReader br = new BufferedReader(new FileReader("data\\msnbc.txt"));
        List<List<List<String>>> transactions = new ArrayList<>();
        try {
            String line = br.readLine();

            while (line != null) {
                String[] items = line.split(" ");

                List<List<String>> transaction = new ArrayList<>();
                for(String item : items) {
                    transaction.add(Arrays.asList(pageMap.get(Integer.parseInt(item))));
                }
                transactions.add(transaction);

                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        JavaRDD<List<List<String>>> sequences = jsc.parallelize(transactions, 8);

        PrefixSpan prefixSpan = new PrefixSpan()
                .setMinSupport(0.0001)
                .setMaxPatternLength(5);
        PrefixSpanModel<String> model = prefixSpan.run(sequences);

        System.out.println("DISPLAY FREQUENT SEQUENCE OF PAGES!!!");
        List<SequentialRule> ruleList = new ArrayList<>();
        Map<List<List<String>>, Double> freqItemsetMap = new HashMap<>();

        Map<String, Long> supportMap = new HashMap<>();
        // display frequent sequences
        for (PrefixSpan.FreqSequence<String> freqSeq: model.freqSequences().toJavaRDD().collect()) {
            //System.out.println(freqSeq.javaSequence() + ", " + freqSeq.freq() + ", " + freqSeq.javaSequence().size());

            freqItemsetMap.put(freqSeq.javaSequence(), freqSeq.freq()*1.0/transactions.size() );

            supportMap.putIfAbsent(freqSeq.javaSequence().toString(), freqSeq.freq());

            // rule generation
            // iteratively increase the consequent size and calculate confidence
            List<List<String>> antecedent = freqSeq.javaSequence();
            int splitIndex = antecedent.size() - 1;
            while (antecedent.size() > 0 && splitIndex > 0) {
                antecedent = antecedent.subList(0, splitIndex--);

                List<List<String>> consequent = new ArrayList<>();
                for (int k = splitIndex+1; k < freqSeq.javaSequence().size(); k++) {
                    consequent.add(freqSeq.javaSequence().get(k));
                }

                long patternSupport = freqSeq.freq();
                long anteSupport = supportMap.get(antecedent.toString());

                double confidence = patternSupport*1.0 / anteSupport;
                double min_conf = 0.5;
                if (confidence > min_conf) {
                    //System.out.println(antecedent + " -> " + consequent + ", " + confidence);
                    ruleList.add(new SequentialRule(antecedent, consequent, confidence, freqSeq.freq()*1.0/transactions.size()));
                }
            }
        }

        System.out.println("********************** GENERATING RULES!!! *************************");
        ruleList.sort(Comparator.comparing(SequentialRule::getConfidence).reversed());
        System.out.println("RULE COUNT = " + ruleList.size());
        // display top-k rules
        System.out.println("RULE, CONFIDENCE, INTEREST, LIFT");
        int topK_rules = 10;
        for (SequentialRule rule : ruleList) {
            if (topK_rules-- == 0) break;
            // Interest(I -> j) = conf(I -> j) - Pr[ j]
            double interest = Math.abs(rule.getConfidence() - freqItemsetMap.get(rule.getConsequent()));

            //double f1_score = 2*rule.getConfidence()*rule.getSupport() / (rule.getConfidence() + rule.getSupport());
            // lift(A->B) = P(A,B) / (P(A)*P(B))
            double lift = rule.getSupport() / (freqItemsetMap.get(rule.getAntecedent()) * freqItemsetMap.get(rule.getConsequent()));
            System.out.println(rule.getAntecedent() + " => " + rule.getConsequent() + ", " + rule.getConfidence()
                    + ", " + interest + ", " + lift);
        }
    }

}
