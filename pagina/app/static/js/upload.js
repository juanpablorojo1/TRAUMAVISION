/* upload.js — Pantalla de carga */
(function () {
  'use strict';

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
  }

  /* Zona de arrastre con vista previa del archivo elegido */
  function setupZone(zone) {
    var input = zone.querySelector('input[type=file]');
    if (!input) return;

    var chosenId = zone.dataset.chosen;
    var chosen = chosenId ? document.getElementById(chosenId) : null;
    var nameEl = chosen ? chosen.querySelector('[data-file-name]') : null;
    var sizeEl = chosen ? chosen.querySelector('[data-file-size]') : null;
    var thumbEl = chosen ? chosen.querySelector('[data-file-thumb]') : null;

    ['dragenter', 'dragover'].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        zone.classList.add('is-dragover');
      });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      zone.addEventListener(ev, function (e) {
        e.preventDefault(); e.stopPropagation();
        if (ev === 'dragleave' && zone.contains(e.relatedTarget)) return;
        zone.classList.remove('is-dragover');
      });
    });
    zone.addEventListener('drop', function (e) {
      if (!e.dataTransfer || !e.dataTransfer.files.length) return;
      input.files = e.dataTransfer.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });

    input.addEventListener('change', function () {
      if (!input.files || !input.files.length || !chosen) return;
      var f = input.files[0];
      if (nameEl) nameEl.textContent = f.name;
      if (sizeEl) sizeEl.textContent = formatSize(f.size);
      chosen.classList.add('is-visible');

      if (thumbEl) {
        var esImagen = f.type && f.type.indexOf('image/') === 0;
        if (esImagen) {
          var url = URL.createObjectURL(f);
          thumbEl.src = url;
          thumbEl.hidden = false;
          thumbEl.onload = function () { URL.revokeObjectURL(url); };
        } else {
          thumbEl.hidden = true;
        }
      }
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-dropzone]').forEach(setupZone);

    /* Selección de región */
    var cards = document.querySelectorAll('[data-region-card]');
    cards.forEach(function (card) {
      card.addEventListener('click', function () {
        if (card.disabled) return;
        var panel = card.dataset.panel;
        document.querySelectorAll('[data-region-card][data-panel="' + panel + '"]')
          .forEach(function (c) { c.setAttribute('aria-pressed', c === card ? 'true' : 'false'); });
        var input = document.getElementById('region-input-' + panel);
        if (input) input.value = card.dataset.region;
      });
    });

    /* Pestañas: imagen suelta o estudio */
    var tabs = document.querySelectorAll('[data-tab]');
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        var target = tab.dataset.tab;
        tabs.forEach(function (t) {
          t.setAttribute('aria-selected', t === tab ? 'true' : 'false');
        });
        document.querySelectorAll('[data-panel-tab]').forEach(function (p) {
          p.hidden = p.dataset.panelTab !== target;
        });
      });
    });

    /* Overlay de progreso mientras corre el modelo */
    var overlay = document.getElementById('progress-overlay');
    if (!overlay) return;
    var titulo = document.getElementById('progress-title');
    var detalle = document.getElementById('progress-detail');
    var segundos = document.getElementById('progress-seconds');
    var placa = overlay.querySelector('[data-scanplate]');
    var thumb = overlay.querySelector('[data-progress-thumb]');
    var cronometro = null;
    var urlPlaca = null;

    function cargarPlaca(file) {
      if (!thumb || !placa) return;
      if (!file || !file.type || file.type.indexOf('image/') !== 0) return;
      urlPlaca = URL.createObjectURL(file);
      thumb.src = urlPlaca;
      thumb.hidden = false;
      placa.classList.add('has-img');
    }

    function mostrar(texto, sub, file) {
      titulo.textContent = texto;
      detalle.textContent = sub;
      cargarPlaca(file);
      overlay.classList.add('open');
      var t = 0;
      segundos.textContent = '0';
      cronometro = setInterval(function () { t += 1; segundos.textContent = String(t); }, 1000);
    }

    var formSingle = document.getElementById('upload-form-single');
    if (formSingle) {
      formSingle.addEventListener('submit', function () {
        var input = document.getElementById('file-input-single');
        mostrar('Analizando la radiografía',
                'El análisis demora unos segundos. No cierre ni recargue esta página.',
                input && input.files.length ? input.files[0] : null);
      });
    }
    var formStudy = document.getElementById('upload-form-study');
    if (formStudy) {
      formStudy.addEventListener('submit', function () {
        var input = document.getElementById('file-input-study');
        var n = input && input.files.length ? input.files[0].name : 'el estudio';
        mostrar('Analizando ' + n,
                'Cada imagen del estudio se procesa por separado; el análisis puede demorar varios minutos.');
      });
    }

    /* Al volver con «atrás» el overlay se cierra */
    window.addEventListener('pageshow', function () {
      overlay.classList.remove('open');
      if (cronometro) { clearInterval(cronometro); cronometro = null; }
      if (urlPlaca) { URL.revokeObjectURL(urlPlaca); urlPlaca = null; }
      if (thumb) { thumb.hidden = true; thumb.removeAttribute('src'); }
      if (placa) placa.classList.remove('has-img');
    });
  });
})();
