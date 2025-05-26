using DigitalHandwriting.Context;
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Models;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public class SettingsService
    {
        private readonly ApplicationContext _context;

        // Define keys for your settings
        private const string ValidationMethodsKey = "ValidationAuthenticationMethods";
        private const string DefaultMethodKey = "DefaultAuthenticationMethod";
        private const string PassphraseInputsKey = "RegistrationPassphraseInputs";

        public SettingsService(ApplicationContext context)
        {
            _context = context;
        }

        public async Task LoadSettingsAsync()
        {
            // Assumes table exists, created by EnsureCreatedAsync on app startup if DB was new.
            var settingsList = await _context.ApplicationSettings.ToListAsync();
            var settings = settingsList.ToDictionary(s => s.Key, s => s.Value);

            if (settings.TryGetValue(ValidationMethodsKey, out var validationMethodsJson))
            {
                try
                {
                    ApplicationConfiguration.ValidationAuthenticationMethods =
                        JsonSerializer.Deserialize<List<Method>>(validationMethodsJson) ?? GetDefaultValidationMethods();
                }
                catch (JsonException)
                {
                    ApplicationConfiguration.ValidationAuthenticationMethods = GetDefaultValidationMethods();
                }
            }
            else
            {
                ApplicationConfiguration.ValidationAuthenticationMethods = GetDefaultValidationMethods();
            }

            if (settings.TryGetValue(DefaultMethodKey, out var defaultMethodStr))
            {
                if (Enum.TryParse<Method>(defaultMethodStr, out var defaultMethod))
                {
                    ApplicationConfiguration.DefaultAuthenticationMethod = defaultMethod;
                }
                else
                {
                    ApplicationConfiguration.DefaultAuthenticationMethod = Method.FilteredManhattan;
                }
            }
            else
            {
                ApplicationConfiguration.DefaultAuthenticationMethod = Method.FilteredManhattan;
            }

            if (settings.TryGetValue(PassphraseInputsKey, out var passphraseInputsStr))
            {
                if (int.TryParse(passphraseInputsStr, out var passphraseInputs) && passphraseInputs > 0)
                {
                    ApplicationConfiguration.RegistrationPassphraseInputs = passphraseInputs;
                }
                else
                {
                    ApplicationConfiguration.RegistrationPassphraseInputs = 3;
                }
            }
            else
            {
                ApplicationConfiguration.RegistrationPassphraseInputs = 3;
            }
        }

        public async Task SaveSettingsAsync()
        {
            // Assumes table exists.
            await UpdateOrCreateSettingAsync(ValidationMethodsKey, JsonSerializer.Serialize(ApplicationConfiguration.ValidationAuthenticationMethods));
            await UpdateOrCreateSettingAsync(DefaultMethodKey, ApplicationConfiguration.DefaultAuthenticationMethod.ToString());
            await UpdateOrCreateSettingAsync(PassphraseInputsKey, ApplicationConfiguration.RegistrationPassphraseInputs.ToString());

            await _context.SaveChangesAsync();
        }

        private async Task UpdateOrCreateSettingAsync(string key, string value)
        {
            var setting = await _context.ApplicationSettings.FindAsync(key);
            if (setting == null)
            {
                _context.ApplicationSettings.Add(new ApplicationSetting { Key = key, Value = value });
            }
            else
            {
                setting.Value = value;
            }
        }

        private List<Method> GetDefaultValidationMethods()
        {
            return new List<Method>
            {
                Method.Euclidian,
                Method.NormalizedEuclidian,
                Method.Manhattan,
                Method.FilteredManhattan,
                Method.ScaledManhattan,
                Method.ITAD,
                Method.CNN,
                Method.GRU,
                Method.KNN,
                Method.LSTM,
                Method.MLP,
                Method.NaiveBayes,
                Method.RandomForest,
                Method.SVM,
                Method.XGBoost
            };
        }
    }
}
