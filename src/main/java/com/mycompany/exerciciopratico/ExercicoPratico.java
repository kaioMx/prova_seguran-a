package com.mycompany.exerciciopratico;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

public class ExercicoPratico{
    public static void main(String args[]) throws Exception {
        Instances datasetTestes = lerDataset("C:\\Users\\kaiom\\OneDrive\\Documents\\NSL-KDD.zip\\NSL-KDD\\KDDTest-21.arff");
        Instances datasetTreinamento = lerDataset("C:\\Users\\kaiom\\OneDrive\\Documents\\NSL-KDD.zip\\NSL-KDD\\KDDTrain+_20Percent.arff");
        J48 classificador = construirJ48(datasetTreinamento);

        int normal = 0;
        int VN = 0;
        int VP = 0;
        int FN = 0;
        int FP = 0;
        
        for (Instance i : datasetTestes) {
            double esperado = i.classValue();
            double resultado = classificador.classifyInstance(i);

            if (resultado == esperado) {
                if (resultado == normal) {
                    VN = VN + 1;
                } else {
                    VP = VP + 1;
                }
                if (resultado == normal) {
                    FN = FN + 1;
                } else {
                    FP = FP + 1;
                }
            }

        }
        float acuracia = Float.valueOf((VP + VN) * 100) / (VP + VN + FP + FN) / 100;
        float recall = Float.valueOf(VP * 100) / (VP + FN) / 100;
        float precision = Float.valueOf((VP * 100) / (VP + FP)) / 100;
        float f1score = (2 * ((recall * precision) / (recall + precision)));
        System.out.println("Resultado Acuracia: "+acuracia);
        System.out.println("Resultado recall: "+recall);
        System.out.println("Resultado precision: "+precision);
        System.out.println("Resultado f1score: "+f1score);
    }
    private static double testarInstancia(J48 classificador, Instance amostra) throws Exception {
        return classificador.classifyInstance(amostra);
    }
    private static J48 construirJ48(Instances treinamento) throws Exception {
        J48 classificador = new J48();
        classificador.buildClassifier(treinamento);
        return classificador;
    }
    public static Instances lerDataset(String dataset) throws IOException {
        FileReader fr = new FileReader(dataset);
        BufferedReader br = new BufferedReader(fr);
        Instances datasetInstances = new Instances(br);
        datasetInstances.setClassIndex(datasetInstances.numAttributes() - 1);
        return datasetInstances;
    }
}