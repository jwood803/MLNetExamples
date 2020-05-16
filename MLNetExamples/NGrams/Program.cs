using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NGrams
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = new List<Input>
            {
                new Input { Text = "I really enjoy being in jazz band." },
                new Input { Text = "But I'm done for the day and am heading home." }
            };

            var dataView = context.Data.LoadFromEnumerable(data);

            var nGramPipeline = context.Transforms.Text.TokenizeIntoWords("Tokens", nameof(Input.Text))
                .Append(context.Transforms.Conversion.MapValueToKey("Tokens")
                .Append(context.Transforms.Text.ProduceNgrams(
                    "NGrams", 
                    "Tokens", 
                    ngramLength: 2, 
                    useAllLengths: false, 
                    weighting: Microsoft.ML.Transforms.Text.NgramExtractingEstimator.WeightingCriteria.Tf)));

            var fitData = nGramPipeline.Fit(dataView);
            var dataTransformed = fitData.Transform(dataView);

            var preview = dataTransformed.Preview();

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            dataTransformed.Schema["NGrams"].GetSlotNames(ref slotNames);

            var nGramsColumn = dataTransformed.GetColumn<VBuffer<float>>(dataTransformed.Schema["NGrams"]);
            var slots = slotNames.GetValues();

            Console.WriteLine("NGrams");

            foreach (var row in nGramsColumn)
            {
                foreach (var item in row.Items())
                {
                    Console.WriteLine($"{slots[item.Key]} ");
                }

                Console.WriteLine();
            }
        }
    }
}
