using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;

namespace CustomVisionOnnx
{
    class Program
    {
        public const int rowCount = 13, columnCount = 13;

        public const int featuresPerBox = 5;

        private static readonly (float x, float y)[] boxAnchors = { (0.573f, 0.677f), (1.87f, 2.06f), (3.34f, 5.47f), (7.88f, 3.53f), (9.77f, 9.17f) };

        private static string[] testFiles = new[] { "./test/red-test.jpg", "./test/white.jpg" };

        static void Main(string[] args)
        {
            Bitmap testImage;
            var context = new MLContext();

            var emptyData = new List<WineInput>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var pipeline = context.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, outputColumnName: "data", imageWidth: ImageSettings.imageWidth, imageHeight: ImageSettings.imageHeight, inputColumnName: nameof(WineInput.Image))
                            .Append(context.Transforms.ExtractPixels(outputColumnName: "data"))
                            .Append(context.Transforms.ApplyOnnxModel(modelFile: "./Model/model.onnx", outputColumnName: "model_outputs0", inputColumnName: "data"));

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<WineInput, WinePredictions>(model);

            var labels = File.ReadAllLines("./Model/labels.txt");

            foreach (var image in testFiles)
            {
                using (var stream = new FileStream(image, FileMode.Open))
                {
                    testImage = (Bitmap)Image.FromStream(stream);
                }

                var prediction = predictionEngine.Predict(new WineInput { Image = testImage });

                var boundingBoxes = ParseOutputs(prediction.PredictedLabels, labels);

                var originalWidth = testImage.Width;
                var originalHeight = testImage.Height;

                if (boundingBoxes.Count > 1)
                {
                    var maxConfidence = boundingBoxes.Max(b => b.Confidence);
                    var topBoundingBox = boundingBoxes.FirstOrDefault(b => b.Confidence == maxConfidence);

                    boundingBoxes.Clear();

                    boundingBoxes.Add(topBoundingBox);
                }

                foreach (var boundingBox in boundingBoxes)
                {
                    float x = Math.Max(boundingBox.Dimensions.X, 0);
                    float y = Math.Max(boundingBox.Dimensions.Y, 0);
                    float width = Math.Min(originalWidth - x, boundingBox.Dimensions.Width);
                    float height = Math.Min(originalHeight - y, boundingBox.Dimensions.Height);

                    // fit to current image size
                    x = originalWidth * x / ImageSettings.imageWidth;
                    y = originalHeight * y / ImageSettings.imageHeight;
                    width = originalWidth * width / ImageSettings.imageWidth;
                    height = originalHeight * height / ImageSettings.imageHeight;

                    using (var graphics = Graphics.FromImage(testImage))
                    {
                        graphics.DrawRectangle(new Pen(Color.Red, 3), x, y, width, height);
                        graphics.DrawString(boundingBox.Description, new Font(FontFamily.Families[0], 30f), Brushes.Red, x + 5, y + 5);
                    }

                    testImage.Save($"{image}-predicted.jpg");
                }
            }
        }

        public static List<BoundingBox> ParseOutputs(float[] modelOutput, string[] labels, float probabilityThreshold = .5f)
        {
            var boxes = new List<BoundingBox>();

            for (int row = 0; row < rowCount; row++)
            {
                for (int column = 0; column < columnCount; column++)
                {
                    for (int box = 0; box < boxAnchors.Length; box++)
                    {
                        var channel = box * (labels.Length + featuresPerBox);

                        var boundingBoxPrediction = ExtractBoundingBoxPrediction(modelOutput, row, column, channel);

                        var mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxPrediction);

                        if (boundingBoxPrediction.Confidence < probabilityThreshold)
                            continue;

                        float[] classProbabilities = ExtractClassProbabilities(modelOutput, row, column, channel, boundingBoxPrediction.Confidence, labels);

                        var (topProbability, topIndex) = classProbabilities.Select((probability, index) => (Score: probability, Index: index)).Max();

                        if (topProbability < probabilityThreshold)
                            continue;

                        boxes.Add(new BoundingBox
                        {
                            Dimensions = mappedBoundingBox,
                            Confidence = topProbability,
                            Label = labels[topIndex]
                        });
                    }
                }
            }

            return boxes;
        }

        private static BoundingBoxDimensions MapBoundingBoxToCell(int row, int column, int box, BoundingBoxPrediction boxDimensions)
        {
            const float cellWidth = ImageSettings.imageWidth / columnCount;
            const float cellHeight = ImageSettings.imageHeight / rowCount;

            var mappedBox = new BoundingBoxDimensions
            {
                X = (row + Sigmoid(boxDimensions.X)) * cellWidth,
                Y = (column + Sigmoid(boxDimensions.Y)) * cellHeight,
                Width = MathF.Exp(boxDimensions.Width) * cellWidth * boxAnchors[box].x,
                Height = MathF.Exp(boxDimensions.Height) * cellHeight * boxAnchors[box].y,
            };

            // The x,y coordinates from the (mapped) bounding box prediction represent the center
            // of the bounding box. We adjust them here to represent the top left corner.
            mappedBox.X -= mappedBox.Width / 2;
            mappedBox.Y -= mappedBox.Height / 2;

            return mappedBox;
        }

        private static BoundingBoxPrediction ExtractBoundingBoxPrediction(float[] modelOutput, int row, int column, int channel)
        {
            return new BoundingBoxPrediction
            {
                X = modelOutput[GetOffset(row, column, channel++)],
                Y = modelOutput[GetOffset(row, column, channel++)],
                Width = modelOutput[GetOffset(row, column, channel++)],
                Height = modelOutput[GetOffset(row, column, channel++)],
                Confidence = Sigmoid(modelOutput[GetOffset(row, column, channel++)])
            };
        }

        public static float[] ExtractClassProbabilities(float[] modelOutput, int row, int column, int channel, float confidence, string[] labels)
        {
            var classProbabilitiesOffset = channel + featuresPerBox;
            float[] classProbabilities = new float[labels.Length];
            for (int classProbability = 0; classProbability < labels.Length; classProbability++)
                classProbabilities[classProbability] = modelOutput[GetOffset(row, column, classProbability + classProbabilitiesOffset)];
            return Softmax(classProbabilities).Select(p => p * confidence).ToArray();
        }

        private static float Sigmoid(float value)
        {
            var k = MathF.Exp(value);
            return k / (1.0f + k);
        }

        private static float[] Softmax(float[] classProbabilities)
        {
            var max = classProbabilities.Max();
            var exp = classProbabilities.Select(v => MathF.Exp(v - max));
            var sum = exp.Sum();
            return exp.Select(v => v / sum).ToArray();
        }

        private static int GetOffset(int row, int column, int channel)
        {
            const int channelStride = rowCount * columnCount;
            return (channel * channelStride) + (column * columnCount) + row;
        }
    }

    class BoundingBoxPrediction : BoundingBoxDimensions
    {
        public float Confidence { get; set; }
    }
}
