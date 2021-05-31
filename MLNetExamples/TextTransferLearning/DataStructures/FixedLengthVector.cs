using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TextTransferLearning.DataStructures
{
    public class FixedLengthVector
    {
        [VectorType(600)]
        public int[] Features { get; set; }
    }
}
