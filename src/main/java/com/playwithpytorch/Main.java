package com.playwithpytorch;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {

    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException {
        File inputFile = new File("/Users/siddharthmudgal/test/image.jpg");
        Image img = ImageFactory.getInstance().fromFile(inputFile.toPath());
        Criteria<Image, DetectedObjects> detectedObjectsCriteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone","resnet50")
                        .build();

        try (ZooModel<Image, DetectedObjects> imageDetectedObjectsZooModel =
                     ModelZoo.loadModel(detectedObjectsCriteria)) {

            try (Predictor<Image, DetectedObjects> objectsPredictor= imageDetectedObjectsZooModel.newPredictor()) {

                DetectedObjects detectedObjects = objectsPredictor.predict(img);
                printDetectedObjectsToDisk(detectedObjects, img);

            } catch (TranslateException e) {
                e.printStackTrace();
            }

        }

    }

    public static void printDetectedObjectsToDisk(DetectedObjects detectedObjects , Image image) throws IOException {

        Path outDir = Paths.get("/Users/siddharthmudgal/test/output.jpeg");

        Image outputImage = image.duplicate(Image.Type.TYPE_INT_ARGB);
        outputImage.drawBoundingBoxes(detectedObjects);
        outputImage.save(Files.newOutputStream(outDir), "png");

    }

}