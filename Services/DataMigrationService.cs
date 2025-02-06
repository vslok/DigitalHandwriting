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
        public double[] FirstDD { get; set; }
        public double[] FirstUD { get; set; }

        public double[] SecondH { get; set; }
        public double[] SecondDD { get; set; }
        public double[] SecondUD { get; set; }

        public double[] ThirdH { get; set; }
        public double[] ThirdDD { get; set; }
        public double[] ThirdUD { get; set; }
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
            var importedUsers = this.ReadUsersFromCsv(filePath);
            foreach (var user in importedUsers)
            {
                _userRepository.AddUser(user);
            }
        }

        public IEnumerable<CsvImportUser> GetUsersFromCsv(string filePath)
        {
            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                PrepareHeaderForMatch = args => args.Header.ToLower(),
            };

            using (var reader = new StreamReader(filePath))
            using (var csv = new CsvReader(reader, config))
            {
                csv.Context.TypeConverterCache.AddConverter<double[]>(new DoubleArrayConverter());
                var records = csv.GetRecords<CsvImportUser>();
                foreach (var record in records)
                {
                    yield return record;
                }
            }
        }

        public List<User> ReadUsersFromCsv(string filePath)
        {
            var records = GetUsersFromCsv(filePath).ToList();
            var users = records.Select(record => new User()
            {
                Login = record.Subject,
                Password = EncryptionService.GetPasswordHash(record.Password, out var salt),
                Salt = salt,
                KeyPressedTimesMedians = JsonSerializer.Serialize(record.ThirdH),
                BetweenKeysTimesMedians = JsonSerializer.Serialize(record.ThirdUD),
                BetweenKeysPressTimesMedians = JsonSerializer.Serialize(record.ThirdDD),
                KeyPressedTimesFirst = JsonSerializer.Serialize(record.FirstH),
                KeyPressedTimesSecond = JsonSerializer.Serialize(record.SecondH),
                KeyPressedTimesThird = JsonSerializer.Serialize(record.ThirdH),
                BetweenKeysTimesFirst = JsonSerializer.Serialize(record.FirstUD),
                BetweenKeysTimesSecond = JsonSerializer.Serialize(record.SecondUD),
                BetweenKeysTimesThird = JsonSerializer.Serialize(record.ThirdUD),
                BetweenKeysPressTimesFirst = JsonSerializer.Serialize(record.FirstDD),
                BetweenKeysPressTimesSecond = JsonSerializer.Serialize(record.SecondDD),
                BetweenKeysPressTimesThird = JsonSerializer.Serialize(record.ThirdDD),
            }).ToList();

            return users;
        }
    }
}
