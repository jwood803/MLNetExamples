using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace BinaryClassification
{
    public class TitanicPrediction
    {
        [ColumnName("Score")]
        public bool WillSurvive;
    }
}
