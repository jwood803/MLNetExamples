using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace BagOfWords
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = new List<Input>
            {
                new Input { Text = "I really enjoy being in jazz band. And I really like it." }
            };

            var dataView = context.Data.LoadFromEnumerable(data);

            var bagWordsPipeline = context.Transforms.Text.ProduceWordBags(
                    "BagOfWords",
                    "Text",
                    ngramLength: 1,
                    useAllLengths: false,
                    weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.Tf
                );

            var bagWordsTransform = bagWordsPipeline.Fit(dataView);
            var bagWordsDataView = bagWordsTransform.Transform(dataView);

            var predictionEngine = context.Model.CreatePredictionEngine<Input, Output>(bagWordsTransform);

            var prediction = predictionEngine.Predict(data[0]);

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            bagWordsDataView.Schema["BagOfWords"].GetSlotNames(ref slotNames);

            var bagOfWordColumn = bagWordsDataView.GetColumn<VBuffer<float>>(bagWordsDataView.Schema["BagOfWords"]);
            var slots = slotNames.GetValues();

            Console.Write("NGrams: ");
            foreach (var featureRow in bagOfWordColumn)
            {
                foreach (var item in featureRow.Items())
                {
                    Console.Write($"{slots[item.Key]}  ");
                }

                Console.WriteLine();
            }

            Console.Write("Features: ");
            for (int i = 0; i < prediction.BagOfWords.Length; i++)
            {
                Console.Write($"{prediction.BagOfWords[i]:F4}  ");
            }

            Console.WriteLine(Environment.NewLine);
        }
    }
}
