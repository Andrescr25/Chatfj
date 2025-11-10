import React, { useState } from 'react';

function DocumentUpload({ API_URL }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadCategory, setUploadCategory] = useState('general');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const categories = [
    { value: 'pension_alimentaria', label: 'Pensión Alimentaria' },
    { value: 'laboral', label: 'Laboral' },
    { value: 'violencia', label: 'Violencia Doméstica' },
    { value: 'civil', label: 'Civil / Desalojos' },
    { value: 'menores', label: 'Menores / PANI' },
    { value: 'penal', label: 'Penal' },
    { value: 'general', label: 'General' }
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validExtensions = ['.pdf', '.txt', '.md'];
      const fileExt = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();

      if (!validExtensions.includes(fileExt)) {
        alert(`Tipo de archivo no válido. Use: ${validExtensions.join(', ')}`);
        return;
      }

      setSelectedFile(file);
      setUploadResult(null);
    }
  };

  const uploadDocument = async () => {
    if (!selectedFile) {
      alert('Por favor selecciona un archivo');
      return;
    }

    setIsUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('category', uploadCategory);

      const response = await fetch(`${API_URL}/training/upload-document?category=${uploadCategory}`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (data.success) {
        setUploadResult({
          success: true,
          message: data.message,
          chunks_added: data.chunks_added,
          total_documents: data.total_documents_in_db
        });
        setSelectedFile(null);
        // Reset file input
        document.getElementById('fileInput').value = '';
      } else {
        setUploadResult({
          success: false,
          message: data.detail || 'Error al subir documento'
        });
      }
    } catch (error) {
      console.error('Error subiendo documento:', error);
      setUploadResult({
        success: false,
        message: 'Error de conexión al subir el documento'
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div style={{
      marginTop: '20px',
      padding: '20px',
      border: '2px solid #2196f3',
      borderRadius: '8px',
      backgroundColor: '#e3f2fd'
    }}>
      <h3 style={{ color: '#1976d2', marginTop: 0 }}>
        📚 Agregar Documentos a la Base de Conocimientos
      </h3>
      <p style={{ fontSize: '14px', color: '#555', marginBottom: '20px' }}>
        Sube documentos oficiales (PDF, TXT, MD) para enriquecer la base vectorial del sistema.
        Los documentos se procesarán automáticamente y estarán disponibles para consultas.
      </p>

      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
          Categoría del documento:
        </label>
        <select
          value={uploadCategory}
          onChange={(e) => setUploadCategory(e.target.value)}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid #2196f3',
            fontSize: '14px'
          }}
        >
          {categories.map(cat => (
            <option key={cat.value} value={cat.value}>{cat.label}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '8px' }}>
          Seleccionar archivo:
        </label>
        <input
          id="fileInput"
          type="file"
          accept=".pdf,.txt,.md"
          onChange={handleFileChange}
          style={{
            width: '100%',
            padding: '10px',
            borderRadius: '4px',
            border: '1px solid #2196f3',
            fontSize: '14px'
          }}
        />
        {selectedFile && (
          <p style={{ marginTop: '8px', color: '#1976d2', fontSize: '13px' }}>
            ✓ Archivo seleccionado: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
          </p>
        )}
      </div>

      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={uploadDocument}
          disabled={!selectedFile || isUploading}
          style={{
            padding: '12px 24px',
            backgroundColor: selectedFile && !isUploading ? '#4caf50' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: selectedFile && !isUploading ? 'pointer' : 'not-allowed',
            fontWeight: 'bold',
            fontSize: '14px'
          }}
        >
          {isUploading ? '⏳ Subiendo...' : '📤 Subir Documento'}
        </button>

        {selectedFile && !isUploading && (
          <button
            onClick={() => {
              setSelectedFile(null);
              setUploadResult(null);
              document.getElementById('fileInput').value = '';
            }}
            style={{
              padding: '12px 24px',
              backgroundColor: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '14px'
            }}
          >
            ✕ Cancelar
          </button>
        )}
      </div>

      {uploadResult && (
        <div style={{
          marginTop: '15px',
          padding: '15px',
          borderRadius: '4px',
          backgroundColor: uploadResult.success ? '#d4edda' : '#f8d7da',
          border: `1px solid ${uploadResult.success ? '#c3e6cb' : '#f5c6cb'}`,
          color: uploadResult.success ? '#155724' : '#721c24'
        }}>
          <strong>{uploadResult.success ? '✅ Éxito!' : '❌ Error'}</strong>
          <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
            {uploadResult.message}
          </p>
          {uploadResult.success && uploadResult.chunks_added && (
            <p style={{ margin: '8px 0 0 0', fontSize: '13px' }}>
              • Fragmentos agregados: {uploadResult.chunks_added}<br/>
              • Total de documentos en DB: {uploadResult.total_documents}
            </p>
          )}
        </div>
      )}

      <div style={{
        marginTop: '15px',
        padding: '12px',
        backgroundColor: '#fff3cd',
        border: '1px solid #ffc107',
        borderRadius: '4px',
        fontSize: '13px',
        color: '#856404'
      }}>
        <strong>💡 Información importante:</strong>
        <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
          <li>Los documentos se dividen automáticamente en fragmentos para mejor búsqueda</li>
          <li>Se crean embeddings vectoriales para búsqueda semántica</li>
          <li>Los documentos están disponibles inmediatamente después de la subida</li>
          <li>Formatos soportados: PDF, TXT, MD</li>
        </ul>
      </div>
    </div>
  );
}

export default DocumentUpload;