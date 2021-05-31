using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using TextTransferLearning.DataStructures;

namespace TextTransferLearning
{
    class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Model");

        static void Main(string[] args)
        {
            var context = new MLContext();

            var tensorFlowModel = context.Model.LoadTensorFlowModel(_modelPath);

            DataViewSchema schema = tensorFlowModel.GetModelSchema();

            Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
            var featuresType = (VectorDataViewType)schema["Features"].Type;

            Console.WriteLine($"Name: Features, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
            var predictionType = (VectorDataViewType)schema["Prediction/Softmax"].Type;

            Console.WriteLine($"Name: Prediction/Softmax, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");

            // Run here to show tensorflow error

            var wordLookup = context.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
                columns: new[]
                   {
                        new TextLoader.Column("Words", DataKind.String, 0),
                        new TextLoader.Column("Ids", DataKind.Int32, 1),
                   },
                separatorChar: ',');

            Action<VariableLengthVector, FixedLengthVector> ResizeFeaturesAction = (s, f) =>
            {
                var features = s.VariableLengthFeatures;
                Array.Resize(ref features, 600);
                f.Features = features;
            };

            var pipeline = context.Transforms.Text.TokenizeIntoWords("Tokens", "Review")
                .Append(context.Transforms.Conversion.MapValue("VariableLengthFeatures", wordLookup, wordLookup.Schema["Words"], wordLookup.Schema["Ids"], "Tokens"))
                .Append(context.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))
                .Append(tensorFlowModel.ScoreTensorFlowModel("Prediction/Softmax", "Features"))
                .Append(context.Transforms.CopyColumns("Prediction", "Prediction/Softmax"));

            var data = context.Data.LoadFromEnumerable(new List<MovieReview>());

            var model = pipeline.Fit(data);

            var predictionEngine = context.Model.CreatePredictionEngine<MovieReview, MovieReviewSentiment>(model);

            var review = new MovieReview
            {
                Review = "I like the action"
            };

            var prediction = predictionEngine.Predict(review);

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine("Number of classes: {0}", prediction.Prediction.Length);
            Console.WriteLine("Sentiment? {0}", prediction.Prediction[1] > 0.5 ? "Positive" : "Negative");

        }
    }
}
