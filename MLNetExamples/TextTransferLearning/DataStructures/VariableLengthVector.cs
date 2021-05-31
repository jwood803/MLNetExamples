using Microsoft.ML.Data;

namespace TextTransferLearning.DataStructures
{
    public class VariableLengthVector
    {
        [VectorType]
        public int[] VariableLengthFeatures { get; set; }
    }
}
