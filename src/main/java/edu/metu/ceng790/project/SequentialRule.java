package edu.metu.ceng790.project;

import java.util.List;

public class SequentialRule {
    private List<List<String>> antecedent;
    private List<List<String>> consequent;
    private double confidence;

    private double support;

    public SequentialRule(List<List<String>> antecedent, List<List<String>> consequent, double confidence, double support) {
        this.antecedent = antecedent;
        this.consequent = consequent;
        this.confidence = confidence;
        this.support = support;
    }

    public double getConfidence() {
        return confidence;
    }

    public List<List<String>> getAntecedent() {
        return antecedent;
    }

    public List<List<String>> getConsequent() {
        return consequent;
    }

    public double getSupport() {
        return support;
    }

    public void setSupport(double support) {
        this.support = support;
    }
}
