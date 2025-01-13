using CsvHelper.Configuration;
using CsvHelper;
using CsvHelper.TypeConversion;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Converters
{
    internal class DoubleArrayConverter : DefaultTypeConverter
    {
        public override object ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            // Remove the square brackets and split by comma
            return text.Trim('[', ']').Split(',').Select(double.Parse).ToArray();
        }
    }
}
