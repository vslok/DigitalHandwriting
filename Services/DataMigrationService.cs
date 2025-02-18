using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using CsvHelper.Configuration;
using CsvHelper;
using DigitalHandwriting.Converters;
using System.Globalization;
using System.Text.Json;

namespace DigitalHandwriting.Services
{
    public class CsvImportUser
    {
        public string Subject { get; set; }
        public int SessionIndex { get; set; }
        public int Rep { get; set; }
        public string Password { get; set; }

        public double[] FirstH { get; set; }
        public double[] FirstDU { get; set; }

        public double[] SecondH { get; set; }
        public double[] SecondDU { get; set; }

        public double[] ThirdH { get; set; }
        public double[] ThirdDU { get; set; }
    }

    public class CsvImportAuthentication
    {
        public string Subject { get; set; }

        public double[] H { get; set; }

        public double[] DU { get; set; }

        public bool IsLegalUser { get; set; }
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
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());
                csv.Read();
                csv.ReadHeader();

                while (csv.Read()) {
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
                csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());
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
            var records = ReadUsersFromCsv(filePath);
            foreach (var record in records)
            {
                yield return new User()
                {
                    Login = record.Subject,
                    Password = EncryptionService.GetPasswordHash(record.Password, out var salt),
                    Salt = salt,
                    FirstH = JsonSerializer.Serialize(record.FirstH),
                    SecondH = JsonSerializer.Serialize(record.SecondH),
                    ThirdH = JsonSerializer.Serialize(record.ThirdH),
                    FirstDU = JsonSerializer.Serialize(record.FirstDU),
                    SecondDU = JsonSerializer.Serialize(record.SecondDU),
                    ThirdDU = JsonSerializer.Serialize(record.ThirdDU),
                };
            }
        }
    }
}
