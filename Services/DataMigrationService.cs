using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using CsvHelper.Configuration;
using CsvHelper;
using DigitalHandwriting.Converters; // This should now include ListOfListOfDoublesConverter
using System.Globalization;
// System.Text.Json might be indirectly used by the converter, but not directly here anymore

namespace DigitalHandwriting.Services
{
    public class CsvImportUser // Updated structure
    {
        public string Login { get; set; }
        public string Password { get; set; }

        // Old properties (FirstH, FirstUD, etc.) are removed.
        // New properties expecting a JSON string in CSV cell representing List<List<double>>
        public List<List<double>> HSampleValues { get; set; }
        public List<List<double>> UDSampleValues { get; set; }
    }

    // CsvImportAuthentication remains unchanged unless its source CSV also changes
    public class CsvImportAuthentication
    {
        public string Login { get; set; }
        public double[] H { get; set; } // Uses DoubleArrayConverter
        public double[] UD { get; set; } // Uses DoubleArrayConverter
        public bool IsLegalUser { get; set; }
    }

    public class CsvExportAuthentication // Unchanged
    {
        public string Login { get; set; }
        public bool IsLegalUser { get; set; }
        public string AuthenticationMethod { get; set; }
        public double Threshold { get; set; }
        public bool IsAuthenticated { get; set; }
        public int N { get; set; }
        public double H_Score { get; set; }
        public double DU_Score { get; set; }
        public double UU_Score { get; set; }
        public double DD_Score { get; set; }
        public double TotalAuthenticationScore { get; set; }
    }

    public class DataMigrationService
    {
        private readonly UserRepository _userRepository;

        public DataMigrationService()
        {
            _userRepository = new UserRepository();
        }

        public void ImportDataFromCsv(string filePath)
        {
            var importedUsers = this.GetUsersFromCsv(filePath);
            foreach (var user in importedUsers)
            {
                _userRepository.AddUser(user);
            }
        }

        public IEnumerable<CsvImportUser> ReadUsersFromCsv(string filePath)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
                // Delimiter = "," // Specify if not comma
                // HasHeaderRecord = true // Default is true
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                // Register the NEW converter for List<List<double>>
                csv.Context.TypeConverterCache.AddConverter<List<List<double>>>(new ListOfListOfDoublesConverter());

                // If CsvImportUser still had any plain double[] properties that need specific conversion,
                // you would keep the DoubleArrayConverter registration for them.
                // csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());

                csv.Read(); // Read the first line
                csv.ReadHeader(); // Read the header row

                while (csv.Read()) { // Read data rows
                    yield return csv.GetRecord<CsvImportUser>();
                }
            }
        }

        public IEnumerable<CsvImportAuthentication> GetAuthenticationDataFromCsv(string filePath)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                // CsvImportAuthentication uses H and UD as double[], so it needs DoubleArrayConverter
                csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());
                csv.Context.TypeConverterCache.AddConverter<List<List<double>>>(new ListOfListOfDoublesConverter()); // Add this if any property in CsvImportAuthentication would use it. Not the case here.


                csv.Read();
                csv.ReadHeader();

                while (csv.Read())
                {
                    yield return csv.GetRecord<CsvImportAuthentication>();
                }
            }
        }

        public IEnumerable<User> GetUsersFromCsv(string filePath)
        {
            var records = ReadUsersFromCsv(filePath); // This now returns CsvImportUser with populated HSampleValues/UDSampleValues
            foreach (var record in records)
            {
                // The CsvImportUser 'record' now has HSampleValues and UDSampleValues as List<List<double>>
                // So, the mapping to the User model is direct.
                yield return new User()
                {
                    Login = record.Login,
                    Password = EncryptionService.GetPasswordHash(record.Password, out var salt), // Assuming EncryptionService is accessible
                    Salt = salt,
                    HSampleValues = record.HSampleValues, // Direct assignment
                    UDSampleValues = record.UDSampleValues  // Direct assignment
                };
            }
        }

        public IEnumerable<CsvExportAuthentication> GetAuthenticationResultsFromCsv(string filePath) // Unchanged
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                csv.Read();
                csv.ReadHeader();

                while (csv.Read())
                {
                    yield return csv.GetRecord<CsvExportAuthentication>();
                }
            }
        }
    }
}
