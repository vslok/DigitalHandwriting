using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.TypeConversion;
using System;
using System.Collections.Generic;
using System.Text.Json; // Using System.Text.Json. Adapt if you use Newtonsoft.Json

namespace DigitalHandwriting.Converters // Ensure this namespace matches your project structure
{
    public class ListOfListOfDoublesConverter : DefaultTypeConverter
    {
        public override object ConvertFromString(string text, IReaderRow row, MemberMapData memberMapData)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                // Return an empty list or null, depending on how your application should treat empty CSV cells.
                // For [Required] properties, an empty list might be preferable to null if the list itself is expected.
                return new List<List<double>>();
            }
            try
            {
                // Assumes the text is a JSON string like "[[1.0,1.1],[2.0,2.1],[3.0,3.1]]"
                var options = new JsonSerializerOptions
                {
                    // Add any specific JsonSerializerOptions if needed
                    PropertyNameCaseInsensitive = true,
                };
                return JsonSerializer.Deserialize<List<List<double>>>(text, options);
            }
            catch (JsonException ex)
            {
                // Log the error or throw a more specific CsvHelper exception
                Console.Error.WriteLine($"Failed to deserialize CSV field to List<List<double>>. Text: '{text}'. Error: {ex.Message}");
                // Depending on strictness, either return an empty list, null, or re-throw.
                // throw new TypeConverterException(this, memberMapData, text, row.Context, $"JSON deserialization error: {ex.Message}", ex);
                return new List<List<double>>(); // Fallback to empty list on error
            }
        }

        public override string ConvertToString(object value, IWriterRow row, MemberMapData memberMapData)
        {
            if (value == null)
            {
                return string.Empty;
            }
            // This is for writing List<List<double>> back to a CSV cell as a JSON string.
            var options = new JsonSerializerOptions
            {
                // WriteIndented = false, // Usually false for CSV cell data
            };
            return JsonSerializer.Serialize(value as List<List<double>>, options);
        }
    }
}
