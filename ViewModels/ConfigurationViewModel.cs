using DigitalHandwriting.Commands;
using DigitalHandwriting.Context;
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models; // For Method enum
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Services; // Added for SettingsService
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel; // Added for PropertyChangedEventArgs
using System.Linq;
using System.Windows.Input;
using System.Threading.Tasks; // Added for Task

namespace DigitalHandwriting.ViewModels
{
    public class MethodSelectionViewModel : BaseViewModel
    {
        public Method Method { get; }
        public string Name => Method.ToString();

        private bool _isSelected;
        public bool IsSelected
        {
            get => _isSelected;
            set => SetProperty(ref _isSelected, value);
        }

        public MethodSelectionViewModel(Method method, bool isSelected)
        {
            Method = method;
            _isSelected = isSelected;
        }
    }

    public class ConfigurationViewModel : BaseViewModel
    {
        private readonly SettingsService _settingsService; // Added

        public ObservableCollection<MethodSelectionViewModel> AllValidationMethods { get; private set; }
        public List<Method> AvailableDefaultMethods => Enum.GetValues(typeof(Method)).Cast<Method>().ToList();

        private Method _selectedDefaultAuthenticationMethod;
        public Method SelectedDefaultAuthenticationMethod
        {
            get => _selectedDefaultAuthenticationMethod;
            set => SetProperty(ref _selectedDefaultAuthenticationMethod, value);
        }

        private int _registrationPassphraseInputs;
        public int RegistrationPassphraseInputs
        {
            get => _registrationPassphraseInputs;
            set
            {
                if (SetProperty(ref _registrationPassphraseInputs, value))
                {
                    (SaveConfigurationCommand as Command)?.RaiseCanExecuteChanged();
                }
            }
        }

        public ICommand SaveConfigurationCommand { get; }
        public ICommand CancelCommand { get; } // To close the window

        public Action CloseAction { get; set; } // Action to close the window

        // Constructor updated to accept SettingsService
        public ConfigurationViewModel(SettingsService settingsService)
        {
            _settingsService = settingsService;
            LoadConfiguration(); // Load current static ApplicationConfiguration values
            SaveConfigurationCommand = new AsyncCommand(SaveChanges, CanSaveChanges); // Changed to AsyncCommand
            CancelCommand = new Command(CancelChanges);
        }

        private void LoadConfiguration()
        {
            var allMethods = Enum.GetValues(typeof(Method)).Cast<Method>();
            var tempValidationMethods = new ObservableCollection<MethodSelectionViewModel>(
                allMethods.Select(m => new MethodSelectionViewModel(m, ApplicationConfiguration.ValidationAuthenticationMethods.Contains(m)))
            );

            // Unsubscribe from old items if any (defensive coding, not strictly needed with current logic but good practice)
            if (AllValidationMethods != null)
            {
                foreach (var oldVm in AllValidationMethods)
                {
                    oldVm.PropertyChanged -= MethodViewModel_PropertyChanged;
                }
            }

            AllValidationMethods = tempValidationMethods;
            foreach (var vm in AllValidationMethods)
            {
                vm.PropertyChanged += MethodViewModel_PropertyChanged;
            }

            SelectedDefaultAuthenticationMethod = ApplicationConfiguration.DefaultAuthenticationMethod;
            _registrationPassphraseInputs = ApplicationConfiguration.RegistrationPassphraseInputs;
            InvokeOnPropertyChangedEvent(nameof(RegistrationPassphraseInputs));
        }

        private void MethodViewModel_PropertyChanged(object sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(MethodSelectionViewModel.IsSelected))
            {
                (SaveConfigurationCommand as AsyncCommand)?.RaiseCanExecuteChanged(); // Adjusted for AsyncCommand
            }
        }

        private bool CanSaveChanges()
        {
            // Basic validation: at least one validation method must be selected
            // and registration inputs must be positive.
            return AllValidationMethods.Any(m => m.IsSelected) && RegistrationPassphraseInputs > 0;
        }

        // Updated to be async and use SettingsService
        private async Task SaveChanges()
        {
            ApplicationConfiguration.ValidationAuthenticationMethods = AllValidationMethods
                .Where(m => m.IsSelected)
                .Select(m => m.Method)
                .ToList();
            ApplicationConfiguration.DefaultAuthenticationMethod = SelectedDefaultAuthenticationMethod;
            ApplicationConfiguration.RegistrationPassphraseInputs = RegistrationPassphraseInputs;

            await _settingsService.SaveSettingsAsync(); // Save to DB

            CloseAction?.Invoke();
        }

        private void CancelChanges()
        {
            CloseAction?.Invoke();
        }

        // Changed back to call InvokeOnPropertyChangedEvent
        protected override void InvokeOnPropertyChangedEvent(string propertyName = null)
        {
            base.InvokeOnPropertyChangedEvent(propertyName);
        }
    }
}
