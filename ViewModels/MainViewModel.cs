using DigitalHandwriting.Stores;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.ViewModels
{
    internal class MainViewModel : BaseViewModel
    {
        private readonly NavigationStore _navigationStore;
        public BaseViewModel CurrentViewModel => _navigationStore.CurrentViewModel;

        public string Title { get; set; }

        public MainViewModel(NavigationStore navigationStore)
        {
            _navigationStore = navigationStore;

            _navigationStore.CurrentViewModelChanged += NavigationStore_CurrentViewModelChanged;

            Title = $"Digital Handwritting";
        }

        private void NavigationStore_CurrentViewModelChanged()
        {
            InvokeOnPropertyChangedEvent(nameof(CurrentViewModel));
        }
    }
}
