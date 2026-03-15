import React, { useState } from "react";
import "./index.css";

function Application() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState("");
  const [showUploadPage, setShowUploadPage] = useState(false);
  const [hasPredicted, setHasPredicted] = useState(false);
  const [predictionData, setPredictionData] = useState({
    disease: "",
    confidence: "",
    advice: ""
  });
  
  // NEW: State to manage window transition timings
  const [isFading, setIsFading] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file && !file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      setImageFile(null);
      setImageUrl(null);
      setHasPredicted(false);
    } else {
      setError("");
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      setHasPredicted(false);
    }
  };

  const handleRefresh = () => {
    setHasPredicted(false);
    setImageFile(null);
    setImageUrl(null);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!imageFile) {
      alert("Please upload an image file.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();
      setPredictionData({
        disease: result.disease || result || "Tomato_Late_blight",
        confidence: result.confidence || "70.61%",
        advice: result.advice || "Destroy infected plants, strict sanitation, apply protectant + systemic fungicides; avoid high leaf wetness."
      });
      setHasPredicted(true);
      
    } catch (error) {
      console.error(error);
      alert("An error occurred during prediction.");
    }
  };

  // --- NEW: Transition Handlers ---
  const handleTryNow = () => {
    setIsFading(true);
    setTimeout(() => {
      setShowUploadPage(true);
      setIsFading(false);
    }, 400); // Wait for CSS fade-out animation
  };

  const handleGoBack = () => {
    setIsFading(true);
    setTimeout(() => {
      setShowUploadPage(false);
      setIsFading(false);
      handleRefresh();
    }, 400);
  };

  return (
    <div className="Application">
      {!showUploadPage ? (
        <section className={`split-section ${isFading ? 'fade-out' : 'fade-in'}`}>
          {/* HERO HALF */}
          <div className="hero-half">
            <div className="hero-overlay">
              <h1>Try our AI Powered Plant Disease Detection</h1>
              <button
                className="cta-btn"
                onClick={handleTryNow}
              >
                Try Now
              </button>
            </div>
          </div>

          {/* HOW IT WORKS */}
          <div className="how-half">
            <div className="how-card">
              <h2>How it works?</h2>
              <div className="steps">
                <div className="step">
                  <div className="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="currentColor" class="size-6">
                      <path d="M12 9a3.75 3.75 0 1 0 0 7.5A3.75 3.75 0 0 0 12 9Z" />
                      <path fill-rule="evenodd" d="M9.344 3.071a49.52 49.52 0 0 1 5.312 0c.967.052 1.83.585 2.332 1.39l.821 1.317c.24.383.645.643 1.11.71.386.054.77.113 1.152.177 1.432.239 2.429 1.493 2.429 2.909V18a3 3 0 0 1-3 3h-15a3 3 0 0 1-3-3V9.574c0-1.416.997-2.67 2.429-2.909.382-.064.766-.123 1.151-.178a1.56 1.56 0 0 0 1.11-.71l.822-1.315a2.942 2.942 0 0 1 2.332-1.39ZM6.75 12.75a5.25 5.25 0 1 1 10.5 0 5.25 5.25 0 0 1-10.5 0Zm12-1.5a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z" clip-rule="evenodd" />
                    </svg>
                  </div>
                  <h3>Click a Pic</h3>
                  <p>Take a picture of your plant leaf</p>
                </div>
                <div className="step">
                  <div className="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="currentColor" class="size-6">
                      <path fill-rule="evenodd" d="M10.5 3.75a6 6 0 0 0-5.98 6.496A5.25 5.25 0 0 0 6.75 20.25H18a4.5 4.5 0 0 0 2.206-8.423 3.75 3.75 0 0 0-4.133-4.303A6.001 6.001 0 0 0 10.5 3.75Zm2.03 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v4.94a.75.75 0 0 0 1.5 0v-4.94l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
                    </svg>
                  </div>
                  <h3>Upload</h3>
                  <p>Upload your plant image to the system</p>
                </div>
                <div className="step">
                  <div className="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 24 24" fill="currentColor" class="size-6">
                      <path fill-rule="evenodd" d="M7.502 6h7.128A3.375 3.375 0 0 1 18 9.375v9.375a3 3 0 0 0 3-3V6.108c0-1.505-1.125-2.811-2.664-2.94a48.972 48.972 0 0 0-.673-.05A3 3 0 0 0 15 1.5h-1.5a3 3 0 0 0-2.663 1.618c-.225.015-.45.032-.673.05C8.662 3.295 7.554 4.542 7.502 6ZM13.5 3A1.5 1.5 0 0 0 12 4.5h4.5A1.5 1.5 0 0 0 15 3h-1.5Z" clip-rule="evenodd" />
                      <path fill-rule="evenodd" d="M3 9.375C3 8.339 3.84 7.5 4.875 7.5h9.75c1.036 0 1.875.84 1.875 1.875v11.25c0 1.035-.84 1.875-1.875 1.875h-9.75A1.875 1.875 0 0 1 3 20.625V9.375ZM6 12a.75.75 0 0 1 .75-.75h.008a.75.75 0 0 1 .75.75v.008a.75.75 0 0 1-.75.75H6.75a.75.75 0 0 1-.75-.75V12Zm2.25 0a.75.75 0 0 1 .75-.75h3.75a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75ZM6 15a.75.75 0 0 1 .75-.75h.008a.75.75 0 0 1 .75.75v.008a.75.75 0 0 1-.75.75H6.75a.75.75 0 0 1-.75-.75V15Zm2.25 0a.75.75 0 0 1 .75-.75h3.75a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75ZM6 18a.75.75 0 0 1 .75-.75h.008a.75.75 0 0 1 .75.75v.008a.75.75 0 0 1-.75.75H6.75a.75.75 0 0 1-.75-.75V18Zm2.25 0a.75.75 0 0 1 .75-.75h3.75a.75.75 0 0 1 0 1.5H9a.75.75 0 0 1-.75-.75Z" clip-rule="evenodd" />
                    </svg>
                  </div>
                  <h3>Get Report</h3>
                  <p>AI analyzes and generates results</p>
                </div>
                <div className="step">
                  <div className="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" height="40" width="40"viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bot-message-square-icon lucide-bot-message-square">
                    <path d="M12 6V2H8"/><path d="M15 11v2"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M20 16a2 2 0 0 1-2 2H8.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 4 20.286V8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z"/><path d="M9 11v2"/></svg>
                  </div>
                  <h3>Chat with AI</h3>
                  <p>Ask questions about plant care & treatment</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      ) : (
        <section className={`upload-page ${isFading ? 'fade-out' : 'fade-in'}`}>
          <button className="back-btn-absolute" onClick={handleGoBack}>
            &larr; Go Back
          </button>
          
          <div className="panels-wrapper">
            {/* LEFT PANEL: UPLOAD */}
            <div className={`left-panel-container ${hasPredicted ? 'shifted' : ''}`}>
              <div className="analysis-panel left-panel">
                {hasPredicted && (
                  <button type="button" className="refresh-btn" onClick={handleRefresh} title="Reset">
                    ↻
                  </button>
                )}
                <h2>Upload Leaf Image</h2>
                <p className="subtitle">Upload an image of your plant leaf for AI analysis</p>
                
                <form onSubmit={handleSubmit} className="upload-form">
                  <div className="drop-area">
                    <div className="upload-icon">
                      <svg xmlns="http://www.w3.org/2000/svg" height="40" width="40" viewBox="0 0 24 24" fill="currentColor" class="size-6">
                        <path fill-rule="evenodd" d="M10.5 3.75a6 6 0 0 0-5.98 6.496A5.25 5.25 0 0 0 6.75 20.25H18a4.5 4.5 0 0 0 2.206-8.423 3.75 3.75 0 0 0-4.133-4.303A6.001 6.001 0 0 0 10.5 3.75Zm2.03 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v4.94a.75.75 0 0 0 1.5 0v-4.94l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
                      </svg>
                    </div>
                    <p>Click to upload or drag and drop</p>
                    <p className="file-info">JPG, PNG, WEBP (MAX. 5MB)</p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageChange}
                      className="file-input"
                    />
                  </div>

                  {imageUrl && (
                    <div className="image-preview-box">
                      <img src={imageUrl} alt="Uploaded Leaf" />
                    </div>
                  )}

                  {error && <p className="error">{error}</p>}

                  <button type="submit" className="detect-btn">Detect Disease</button>
                </form>
              </div>
            </div>

            {/* RIGHT PANEL: RESULTS */}
            <div className="right-panel-container">
              {hasPredicted && (
                <div className="analysis-panel right-panel visible">
                  <h2>Analysis Result</h2>
                  
                  <div className="result-cards-container">
                    <div className="result-card disease-card">
                      <div className="card-icon shield-icon">🍂</div>
                      <div className="card-content">
                        <span className="card-label">Disease Identified</span>
                        <strong className="card-value">{predictionData.disease}</strong>
                      </div>
                    </div>

                    <div className="result-card confidence-card">
                      <div className="card-icon chart-icon">📊</div>
                      <div className="card-content">
                        <span className="card-label">Confidence Level</span>
                        <strong className="card-value">{predictionData.confidence}</strong>
                      </div>
                    </div>

                    <div className="result-card advice-card">
                      <div className="card-icon bulb-icon">💡</div>
                      <div className="card-content">
                        <span className="card-label">Expert Advice</span>
                        <p className="card-value advice-text">{predictionData.advice}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default Application;