using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;

namespace WordEmbeddings
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var emptyData = context.Data.LoadFromEnumerable(new List<TextInput>());

            var embeddingsPipline = context.Transforms.Text.NormalizeText("Text", null, keepDiacritics: false, keepPunctuations: false, keepNumbers: false)
                .Append(context.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(context.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding));

            var embeddingTransformer = embeddingsPipline.Fit(emptyData);

            var predictionEngine = context.Model.CreatePredictionEngine<TextInput, TextFeatures>(embeddingTransformer);

            var newData = new TextInput { Text = "No!" };

            var prediction = predictionEngine.Predict(newData);

            Console.WriteLine($"Number of Features: {prediction.Features.Length}");

            // Print the embedding vector.
            Console.WriteLine("Features: ");
            foreach (var feature in prediction.Features)
            {
                Console.Write($"{feature:F4} ");
            }
        }
    }
}
