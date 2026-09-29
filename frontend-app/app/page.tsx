"use client";

import { useEffect, useRef, useState } from "react";

type QualityDetails = {
  passed: boolean;
  reason: string;
  brightness?: number;
  sharpness?: number;
};

type PredictionResponse = {
  analysis_type?: string;
  quality_passed: boolean;
  quality_message: string;
  quality_details: QualityDetails;

  prediction?: string;
  predicted_code?: string;
  predicted_class?: string;
  description?: string;
  guidance?: string;
  confidence?: number | null;
  uncertain?: boolean;
  probabilities?: Record<string, number>;
  gradcam_image?: string;

  supported_categories?: string[];
  disclaimer?: string;
  model_used?: string | null;
  candidate_outputs?: Record<
    string,
    {
      highest_category?: string;
      model_score?: number;
    }
  >;
};

type BroadCandidate = {
  rank: number;
  class_index: number;
  condition: string;
  score: number;
};

type BroadResponse = {
  analysis_type?: string;
  model_used?: string;
  prediction?: string;
  top_5?: BroadCandidate[];
  supported_categories?: number;
  quality_passed: boolean;
  quality_message: string;
  quality_details: QualityDetails;
  experimental?: boolean;
  disclaimer?: string;
};

export default function AnalysisPage() {
  const [file, setFile] =
    useState<File | null>(null);

  const [preview, setPreview] =
    useState<string | null>(null);

  const [result, setResult] =
    useState<PredictionResponse | null>(null);

  const [broadResult, setBroadResult] =
    useState<BroadResponse | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [broadLoading, setBroadLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [cameraActive, setCameraActive] =
    useState(false);

  const [facingMode, setFacingMode] =
    useState<"user" | "environment">("environment");

  const [cameraCount, setCameraCount] =
    useState(0);

  const videoRef =
    useRef<HTMLVideoElement | null>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  // ======================================================
  // CAMERA DETECTION
  // ======================================================

  useEffect(() => {
    async function detectCameras() {
      try {
        const devices =
          await navigator.mediaDevices.enumerateDevices();

        const cameras =
          devices.filter(
            (device) =>
              device.kind === "videoinput"
          );

        setCameraCount(cameras.length);
      } catch (err) {
        console.error(
          "Camera detection failed:",
          err
        );
      }
    }

    detectCameras();

    return () => {
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) =>
            track.stop()
          );
      }
    };
  }, []);

  // ======================================================
  // CAMERA
  // ======================================================

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) =>
          track.stop()
        );

      streamRef.current = null;
    }

    setCameraActive(false);
  }

  async function startCamera(
    mode = facingMode
  ) {
    try {
      setError("");
      setResult(null);
      setBroadResult(null);

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) =>
            track.stop()
          );
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: {
              ideal: mode,
            },
          },
          audio: false,
        });

      streamRef.current = stream;
      setCameraActive(true);

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject =
            stream;
        }
      }, 100);

      const devices =
        await navigator.mediaDevices.enumerateDevices();

      const cameras =
        devices.filter(
          (device) =>
            device.kind === "videoinput"
        );

      setCameraCount(cameras.length);
    } catch (err) {
      console.error(err);

      setError(
        "Camera access failed. Please allow camera permission or upload an image instead."
      );
    }
  }

  async function switchCamera() {
    const nextMode =
      facingMode === "environment"
        ? "user"
        : "environment";

    setFacingMode(nextMode);

    await startCamera(nextMode);
  }

  function captureImage() {
    if (!videoRef.current) {
      return;
    }

    const video =
      videoRef.current;

    const canvas =
      document.createElement("canvas");

    canvas.width =
      video.videoWidth;

    canvas.height =
      video.videoHeight;

    const context =
      canvas.getContext("2d");

    if (!context) {
      return;
    }

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          return;
        }

        const capturedFile =
          new File(
            [blob],
            "dermawise-camera.jpg",
            {
              type: "image/jpeg",
            }
          );

        if (preview) {
          URL.revokeObjectURL(preview);
        }

        setFile(capturedFile);
        setPreview(
          URL.createObjectURL(blob)
        );

        setResult(null);
        setBroadResult(null);
        setError("");

        stopCamera();
      },
      "image/jpeg",
      0.95
    );
  }

  // ======================================================
  // FILE UPLOAD
  // ======================================================

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const selected =
      event.target.files?.[0];

    if (!selected) {
      return;
    }

    stopCamera();

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(selected);

    setPreview(
      URL.createObjectURL(selected)
    );

    setResult(null);
    setBroadResult(null);
    setError("");
  }

  // ======================================================
  // ANALYSIS
  // ======================================================

  async function handleAnalyze() {
    if (!file) {
      setError(
        "Please upload or capture a skin image first."
      );

      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setBroadResult(null);

    try {
      const formData =
        new FormData();

      formData.append(
        "file",
        file
      );

      const endpoint =
        "http://127.0.0.1:8001/analyze";

      const response =
        await fetch(
          endpoint,
          {
            method: "POST",
            body: formData,
          }
        );

      if (!response.ok) {
        throw new Error(
          "Analysis request failed."
        );
      }

      const data:
        PredictionResponse =
        await response.json();

      if (
        data.quality_passed === false
      ) {
        setError(
          data.quality_message
        );

        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "DermaWise could not analyze the image. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  // ======================================================
  // EXPERIMENTAL 114-CONDITION ANALYSIS
  // ======================================================

  async function handleBroadAnalyze() {
    if (!file) {
      setError(
        "Please upload or capture a skin image first."
      );
      return;
    }

    setBroadLoading(true);
    setError("");
    setResult(null);
    setBroadResult(null);

    try {
      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      const response =
        await fetch(
          "http://127.0.0.1:8001/predict-broad",
          {
            method: "POST",
            body: formData,
          }
        );

      if (!response.ok) {
        throw new Error(
          "Experimental analysis request failed."
        );
      }

      const data:
        BroadResponse =
        await response.json();

      if (
        data.quality_passed === false
      ) {
        setError(
          data.quality_message
        );
        return;
      }

      setBroadResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "DermaWise could not run the experimental 114-condition analysis. Make sure the backend is running."
      );
    } finally {
      setBroadLoading(false);
    }
  }

  // ======================================================
  // RESET
  // ======================================================

  function resetAnalysis() {
    stopCamera();

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setBroadResult(null);
    setError("");
    setLoading(false);
    setBroadLoading(false);
  }

  // ======================================================
  // UI
  // ======================================================

  return (
    <main className="min-h-screen bg-slate-50">

      {/* NAVIGATION */}

      <nav className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">

          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 font-bold text-white">
              D
            </div>

            <div>
              <p className="font-bold text-slate-900">
                DermaWise
              </p>

              <p className="text-xs text-slate-500">
                AI Skin Analysis
              </p>
            </div>
          </div>

          <div className="hidden gap-6 text-sm font-medium text-slate-600 sm:flex">
            <a
              href="#analyze"
              className="hover:text-blue-600"
            >
              Analyze
            </a>

            <a
              href="#technology"
              className="hover:text-blue-600"
            >
              About
            </a>

            <a
              href="#safety"
              className="hover:text-blue-600"
            >
              Safety
            </a>
          </div>

        </div>
      </nav>

      {/* HERO */}

      <section className="bg-gradient-to-b from-blue-50 to-slate-50 px-5 py-14">
        <div className="mx-auto max-w-4xl text-center">

          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600 text-3xl text-white shadow-lg">
            ✦
          </div>

          <p className="mt-6 text-sm font-semibold uppercase tracking-widest text-blue-600">
            DermaWise AI
          </p>

          <h1 className="mt-3 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
            Simple skin insights, powered by AI
          </h1>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-600">
            Take or upload a clear photo and get an easy-to-understand insight for the skin concerns DermaWise currently supports.
          </p>

          <div className="mt-7 flex flex-wrap justify-center gap-3 text-sm">

            <span className="rounded-full border border-slate-200 bg-white px-4 py-2 text-slate-600 shadow-sm">
              📷 Upload or take a photo
            </span>

            <span className="rounded-full border border-slate-200 bg-white px-4 py-2 text-slate-600 shadow-sm">
              ✨ Easy-to-read insights
            </span>

            <span className="rounded-full border border-slate-200 bg-white px-4 py-2 text-slate-600 shadow-sm">
              💡 Helpful next steps
            </span>

          </div>
        </div>
      </section>

      {/* ANALYZER */}

      <section
        id="analyze"
        className="px-4 pb-16"
      >
        <div className="mx-auto max-w-4xl rounded-3xl border border-slate-200 bg-white p-5 shadow-xl sm:p-8">

          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-100 text-xl">
              🔍
            </div>

            <div>
              <h2 className="text-2xl font-bold text-slate-900">
                Check your skin photo
              </h2>

              <p className="text-sm text-slate-500">
                Choose a clear, well-lit photo. DermaWise will take care of the rest.
              </p>
            </div>
          </div>

          {/* UNIFIED ANALYSIS */}

          <div className="mt-6 rounded-2xl border border-blue-100 bg-blue-50 p-4">
            <p className="font-semibold text-blue-900">
              No complicated choices
            </p>
            <p className="mt-1 text-sm leading-6 text-blue-800">
              Just add your photo and tap Analyze. You don’t need to choose a model or understand any technical settings.
            </p>
          </div>

          {/* UPLOAD */}

          <div className="mt-7 rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5 sm:p-7">

            <div className="text-center">
              <div className="text-3xl">
                📷
              </div>

              <h3 className="mt-2 font-bold text-slate-900">
                Add your photo
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                For the clearest result, use good lighting and avoid filters.
              </p>
            </div>

            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              className="mt-5 w-full rounded-xl border border-slate-300 bg-white p-3 text-sm"
            />

            <div className="my-4 flex items-center gap-3">
              <div className="h-px flex-1 bg-slate-200" />

              <span className="text-xs text-slate-400">
                OR
              </span>

              <div className="h-px flex-1 bg-slate-200" />
            </div>

            {!cameraActive && (
              <button
                type="button"
                onClick={() =>
                  startCamera()
                }
                className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 font-semibold text-slate-700 transition hover:bg-slate-100"
              >
                📷 Take a Photo
              </button>
            )}

            {cameraActive && (
              <div className="mt-4">

                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full rounded-2xl bg-black"
                />

                <div
                  className={`mt-4 grid gap-3 ${
                    cameraCount > 1
                      ? "sm:grid-cols-3"
                      : "sm:grid-cols-2"
                  }`}
                >
                  <button
                    type="button"
                    onClick={captureImage}
                    className="rounded-xl bg-blue-600 px-4 py-3 font-semibold text-white"
                  >
                    Capture
                  </button>

                  {cameraCount > 1 && (
                    <button
                      type="button"
                      onClick={switchCamera}
                      className="rounded-xl border border-slate-300 bg-white px-4 py-3 font-semibold text-slate-700"
                    >
                      Switch Camera
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={stopCamera}
                    className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 font-semibold text-red-700"
                  >
                    Cancel
                  </button>
                </div>

              </div>
            )}

            {/* PREVIEW */}

            {preview &&
              !cameraActive && (
                <div className="mt-5">

                  <p className="mb-2 text-sm font-medium text-slate-600">
                    Your photo is ready
                  </p>

                  <img
                    src={preview}
                    alt="Skin image selected for analysis"
                    className="mx-auto max-h-96 w-full rounded-2xl border border-slate-200 object-contain"
                  />

                </div>
              )}

          </div>

          {/* ANALYZE */}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={
              !file ||
              loading ||
              broadLoading
            }
            className="mt-5 w-full rounded-xl bg-blue-600 px-5 py-4 text-lg font-bold text-white shadow-md transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? "Analyzing your image..."
              : "✨ Analyze My Photo"}
          </button>

          <div className="mt-4 rounded-2xl border border-violet-200 bg-violet-50 p-4">
            <p className="font-semibold text-violet-900">
              Want to explore more categories?
            </p>

            <p className="mt-1 text-sm leading-6 text-violet-800">
              Experimental research mode compares the photo across 114 skin-condition categories and shows the Top 5 model matches.
            </p>

            <button
              type="button"
              onClick={handleBroadAnalyze}
              disabled={
                !file ||
                loading ||
                broadLoading
              }
              className="mt-3 w-full rounded-xl border border-violet-300 bg-white px-5 py-3 font-bold text-violet-800 transition hover:bg-violet-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {broadLoading
                ? "Exploring 114 conditions..."
                : "🧪 Explore 114 Conditions — Experimental"}
            </button>
          </div>

          <p className="mt-3 text-center text-xs leading-5 text-slate-400">
            Research prototype. DermaWise
            does not provide a medical diagnosis.
          </p>

          {/* LOADING */}

          {(loading || broadLoading) && (
            <div className="mt-6 rounded-2xl border border-blue-200 bg-blue-50 p-6 text-center">

              <div className="mx-auto h-10 w-10 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />

              <p className="mt-4 font-bold text-blue-900">
                {broadLoading
                  ? "Exploring research categories…"
                  : "Looking at your photo…"}
              </p>

              <p className="mt-1 text-sm text-blue-700">
                {broadLoading
                  ? "Comparing the photo across 114 experimental categories. This may take a moment."
                  : "Checking photo quality and visible patterns. This should only take a moment."}
              </p>

            </div>
          )}

          {/* ERROR */}

          {error && (
            <div className="mt-6 rounded-2xl border border-red-200 bg-red-50 p-5">

              <p className="font-bold text-red-800">
                We couldn’t read this photo
              </p>

              <p className="mt-1 text-sm leading-6 text-red-700">
                {error}
              </p>

            </div>
          )}

          {/* EXPERIMENTAL 114-CONDITION RESULT */}

          {broadResult && (
            <div className="mt-9">

              <div className="border-t border-slate-200 pt-8">

                <div className="flex flex-wrap items-center justify-between gap-3">

                  <div>
                    <p className="text-sm font-semibold text-violet-600">
                      EXPERIMENTAL RESEARCH ANALYSIS
                    </p>

                    <h2 className="mt-1 text-3xl font-bold text-slate-900">
                      Top 5 model matches
                    </h2>
                  </div>

                  <span className="rounded-full bg-violet-100 px-3 py-1 text-xs font-semibold text-violet-700">
                    114 categories
                  </span>

                </div>

                <div className="mt-6 rounded-3xl border border-violet-200 bg-violet-50 p-5 sm:p-6">
                  <p className="text-sm leading-6 text-violet-900">
                    This research mode ranks the categories that received the highest model scores. These scores are not clinical probabilities and the first result is not a diagnosis.
                  </p>
                </div>

                <div className="mt-5 space-y-3">

                  {(broadResult.top_5 ?? []).map(
                    (candidate) => (
                      <div
                        key={`${candidate.rank}-${candidate.class_index}`}
                        className="rounded-2xl border border-slate-200 bg-white p-4"
                      >

                        <div className="flex items-center justify-between gap-4">

                          <div className="flex min-w-0 items-center gap-3">
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-violet-100 font-bold text-violet-700">
                              {candidate.rank}
                            </div>

                            <p className="font-semibold capitalize text-slate-800">
                              {candidate.condition}
                            </p>
                          </div>

                          <p className="shrink-0 font-bold text-slate-900">
                            {candidate.score}%
                          </p>

                        </div>

                        <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-violet-600"
                            style={{
                              width: `${Math.min(
                                candidate.score,
                                100
                              )}%`,
                            }}
                          />
                        </div>

                      </div>
                    )
                  )}

                </div>

                <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-5">

                  <p className="font-bold text-amber-900">
                    Experimental mode
                  </p>

                  <p className="mt-2 text-sm leading-6 text-amber-800">
                    {broadResult.disclaimer ??
                      "Experimental research output only. These model rankings should not be interpreted as a medical diagnosis."}
                  </p>

                </div>

                <button
                  type="button"
                  onClick={resetAnalysis}
                  className="mt-6 w-full rounded-xl bg-slate-900 px-5 py-4 font-bold text-white transition hover:bg-slate-800"
                >
                  Analyze Another Image
                </button>

              </div>
            </div>
          )}

          {/* RESULT */}

          {result && (
            <div className="mt-9">

              <div className="border-t border-slate-200 pt-8">

                <div className="flex flex-wrap items-center justify-between gap-3">

                  <div>
                    <p className="text-sm font-semibold text-blue-600">
                      YOUR SKIN INSIGHT
                    </p>

                    <h2 className="mt-1 text-3xl font-bold text-slate-900">
                      Here’s what we found
                    </h2>
                  </div>

                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                    Research Prototype
                  </span>

                </div>

                {/* MAIN RESULT */}

                <div
                  className={`mt-6 rounded-3xl border p-6 ${
                    result.uncertain
                      ? "border-amber-200 bg-amber-50"
                      : "border-blue-200 bg-blue-50"
                  }`}
                >

                  <p className="text-sm font-medium text-slate-600">
                    {result.uncertain
                      ? "Highest-scoring finding"
                      : "Possible match"}
                  </p>

                  <p className="mt-2 text-3xl font-bold text-slate-900">
                    {result.uncertain
                      ? result.predicted_class
                      : result.prediction}
                  </p>

                  <div className="mt-5 rounded-2xl bg-white p-4">

                    <div className="flex items-end justify-between gap-4">
                      <div>
                        <p className="text-sm text-slate-500">
                          Match score
                        </p>

                        <p className="mt-1 text-3xl font-bold text-slate-900">
                          {result.confidence}%
                        </p>
                      </div>

                      <div
                        className={`rounded-full px-3 py-1 text-xs font-bold ${
                          result.uncertain
                            ? "bg-amber-100 text-amber-800"
                            : "bg-emerald-100 text-emerald-700"
                        }`}
                      >
                        {result.uncertain
                          ? "Uncertain"
                          : "Clearer match"}
                      </div>
                    </div>

                    <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full rounded-full bg-blue-600"
                        style={{
                          width: `${Math.min(
                            result.confidence ??
                              0,
                            100
                          )}%`,
                        }}
                      />
                    </div>

                  </div>

                </div>

                {/* UNCERTAINTY */}

                {result.uncertain && (
                  <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-5">

                    <p className="font-bold text-amber-900">
                      ⚠ Uncertain result
                    </p>

                    <p className="mt-2 text-sm leading-6 text-amber-800">
                      The highest model score is
                      below the current 60%
                      prototype threshold. The
                      category is shown for
                      transparency and should not
                      be interpreted as a diagnosis.
                    </p>

                  </div>
                )}

                {/* DESCRIPTION */}

                {(result.description ||
                  result.guidance) && (
                  <div className="mt-6 grid gap-4 sm:grid-cols-2">

                    <div className="rounded-2xl border border-slate-200 p-5">

                      <p className="text-sm font-semibold text-blue-600">
                        WHAT THIS MAY MEAN
                      </p>

                      <p className="mt-3 leading-7 text-slate-700">
                        {result.description}
                      </p>

                    </div>

                    <div className="rounded-2xl border border-slate-200 p-5">

                      <p className="text-sm font-semibold text-blue-600">
                        WHAT YOU CAN DO
                      </p>

                      <p className="mt-3 leading-7 text-slate-700">
                        {result.guidance}
                      </p>

                    </div>

                  </div>
                )}

                {/* GRADCAM */}

                {result.gradcam_image && (
                  <div className="mt-7 rounded-3xl border border-slate-200 p-5 sm:p-6">

                    <div>
                      <p className="text-sm font-semibold text-blue-600">
                        WHY THE AI NOTICED THIS
                      </p>

                      <h3 className="mt-1 text-xl font-bold text-slate-900">
                        Areas the AI focused on
                      </h3>

                      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                        This view highlights areas that influenced the AI result. It does not show a confirmed problem area.
                      </p>
                    </div>

                    <div className="mt-5 grid gap-5 sm:grid-cols-2">

                      <div>
                        <p className="mb-2 text-center text-sm font-semibold text-slate-600">
                          Original
                        </p>

                        {preview && (
                          <img
                            src={preview}
                            alt="Original skin image"
                            className="w-full rounded-2xl border border-slate-200 object-contain"
                          />
                        )}
                      </div>

                      <div>
                        <p className="mb-2 text-center text-sm font-semibold text-slate-600">
                          AI focus
                        </p>

                        <img
                          src={
                            result.gradcam_image
                          }
                          alt="Grad-CAM attention map"
                          className="w-full rounded-2xl border border-slate-200 object-contain"
                        />
                      </div>

                    </div>

                    <div className="mt-4 rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">
                      The highlighted areas explain what influenced the AI, not whether an area is medically abnormal.
                    </div>

                  </div>
                )}

                {/* SCORES */}

                <details className="mt-7 rounded-2xl border border-slate-200 bg-white">
                  <summary className="cursor-pointer list-none p-5 font-semibold text-slate-700">
                    See technical score details
                  </summary>
                  <div className="px-5 pb-5">
<div className="rounded-2xl border border-slate-100 p-5">

                  <p className="text-sm font-semibold text-blue-600">
                    TECHNICAL DETAILS
                  </p>

                  <h3 className="mt-1 text-xl font-bold text-slate-900">
                    Category scores
                  </h3>

                  <p className="mt-2 text-sm text-slate-500">
                    These are relative AI model scores, not medical probabilities.
                  </p>

                  <div className="mt-6 space-y-5">

                    {Object.entries(
                      result.probabilities ??
                        {}
                    )
                      .sort(
                        (a, b) =>
                          b[1] - a[1]
                      )
                      .map(
                        ([
                          name,
                          score,
                        ]) => (
                          <div key={name}>

                            <div className="mb-2 flex justify-between gap-4 text-sm">
                              <span className="font-medium text-slate-700">
                                {name}
                              </span>

                              <span className="font-bold text-slate-900">
                                {score}%
                              </span>
                            </div>

                            <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">

                              <div
                                className="h-full rounded-full bg-blue-600"
                                style={{
                                  width: `${Math.min(
                                    score,
                                    100
                                  )}%`,
                                }}
                              />

                            </div>

                          </div>
                        )
                      )}

                  </div>

                </div>

                                  </div>
                </details>

                {/* DISCLAIMER */}

                <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-5">

                  <p className="font-semibold text-slate-800">
                    A quick note
                  </p>

                  <p className="mt-2 text-sm leading-6 text-slate-600">
                    DermaWise is an educational AI research prototype, not a diagnosis. If you are worried about a skin change or symptoms persist, speak with a qualified healthcare professional.
                  </p>

                </div>

                <button
                  type="button"
                  onClick={resetAnalysis}
                  className="mt-6 w-full rounded-xl bg-slate-900 px-5 py-4 font-bold text-white transition hover:bg-slate-800"
                >
                  Analyze Another Image
                </button>

              </div>
            </div>
          )}

        </div>
      </section>

      {/* TECHNOLOGY */}

      <section
        id="technology"
        className="border-t border-slate-200 bg-white px-5 py-16"
      >
        <div className="mx-auto max-w-5xl">

          <div className="text-center">
            <p className="text-sm font-semibold uppercase tracking-widest text-blue-600">
              About DermaWise
            </p>

            <h2 className="mt-2 text-3xl font-bold text-slate-900">
              How DermaWise works
            </h2>
          </div>

          <div className="mt-9 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">

            {[
              [
                "🧠",
                "Specialist AI",
                "DermaWise uses trained specialist models behind one simple photo-analysis experience."
              ],
              [
                "🔎",
                "Visual explanation",
                "An optional attention view shows which parts of an image influenced the AI output."
              ],
              [
                "✓",
                "Photo check",
                "The app checks lighting and sharpness before trying to analyze a photo."
              ],
              [
                "⚠",
                "Knows when to be unsure",
                "When the supported models do not give one clear result, DermaWise returns an uncertain result instead of forcing a category."
              ],
            ].map(
              ([
                icon,
                title,
                text,
              ]) => (
                <div
                  key={title}
                  className="rounded-2xl border border-slate-200 p-5"
                >
                  <div className="text-2xl">
                    {icon}
                  </div>

                  <p className="mt-3 font-bold text-slate-900">
                    {title}
                  </p>

                  <p className="mt-2 text-sm leading-6 text-slate-500">
                    {text}
                  </p>
                </div>
              )
            )}

          </div>
        </div>
      </section>

      {/* RESEARCH / PERFORMANCE */}

      <section className="bg-slate-50 px-5 py-16">
        <div className="mx-auto max-w-5xl">

          <p className="text-sm font-semibold uppercase tracking-widest text-blue-600">
            For those who want the details
          </p>

          <h2 className="mt-2 text-3xl font-bold text-slate-900">
            Technical evaluation
          </h2>

          <p className="mt-3 max-w-3xl leading-7 text-slate-600">
            These details are for project reviewers and anyone interested in how the prototype was evaluated. They are not needed to use the analyzer.
          </p>

          <div className="mt-8 grid gap-5 md:grid-cols-2">

            <div className="rounded-3xl border border-slate-200 bg-white p-6">

              <p className="text-sm font-semibold text-blue-600">
                INTERNAL TESTING
              </p>

              <h3 className="mt-2 text-xl font-bold text-slate-900">
                Lesion Model
              </h3>

              <p className="mt-4 text-4xl font-bold text-slate-900">
                71.67%
              </p>

              <p className="mt-1 text-sm text-slate-500">
                HAM10000 group-aware test
                accuracy
              </p>

              <p className="mt-5 text-sm leading-6 text-slate-600">
                The split was performed by
                lesion ID to prevent the same
                lesion appearing across training
                and test groups.
              </p>

            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6">

              <p className="text-sm font-semibold text-violet-600">
                EXTERNAL EVALUATION
              </p>

              <h3 className="mt-2 text-xl font-bold text-slate-900">
                Fitzpatrick17k
              </h3>

              <p className="mt-4 text-4xl font-bold text-slate-900">
                363
              </p>

              <p className="mt-1 text-sm text-slate-500">
                compatible evaluable images
              </p>

              <p className="mt-5 text-sm leading-6 text-slate-600">
                Used to examine external
                generalization across available
                Fitzpatrick skin-type groups.
                This is not a Fitzpatrick-type
                prediction feature.
              </p>

            </div>

          </div>

          {/* FITZPATRICK */}

          <div className="mt-5 rounded-3xl border border-slate-200 bg-white p-6">

            <h3 className="text-xl font-bold text-slate-900">
              Fitzpatrick17k subgroup results
            </h3>

            <p className="mt-2 text-sm leading-6 text-slate-500">
              External subgroup accuracy.
              Sample sizes differ substantially,
              so these results should not be
              interpreted as definitive fairness
              measurements.
            </p>

            <div className="mt-6 space-y-5">

              {[
                ["Type I", 34.48, 29],
                ["Type II", 26.97, 89],
                ["Type III", 23.64, 110],
                ["Type IV", 15.56, 90],
                ["Type V", 21.62, 37],
                ["Type VI", 25.0, 8],
              ].map(
                ([
                  type,
                  accuracy,
                  count,
                ]) => (
                  <div key={String(type)}>

                    <div className="mb-2 flex justify-between text-sm">
                      <span className="font-medium text-slate-700">
                        {type}{" "}
                        <span className="text-slate-400">
                          n={count}
                        </span>
                      </span>

                      <span className="font-bold text-slate-900">
                        {Number(
                          accuracy
                        ).toFixed(2)}
                        %
                      </span>
                    </div>

                    <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">

                      <div
                        className="h-full rounded-full bg-violet-600"
                        style={{
                          width: `${accuracy}%`,
                        }}
                      />

                    </div>

                  </div>
                )
              )}

            </div>

            <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-5">

              <p className="font-bold text-amber-900">
                Evaluation limitation
              </p>

              <p className="mt-2 text-sm leading-6 text-amber-800">
                The external subset was small
                and imbalanced, some lesion
                categories were unavailable,
                and the image domain differs
                from HAM10000. These results
                demonstrate a generalization
                limitation rather than proving
                skin-tone bias.
              </p>

            </div>

          </div>

        </div>
      </section>

      {/* SAFETY */}

      <section
        id="safety"
        className="border-t border-slate-200 bg-white px-5 py-14"
      >
        <div className="mx-auto max-w-4xl text-center">

          <div className="text-3xl">
            🛡️
          </div>

          <h2 className="mt-3 text-2xl font-bold text-slate-900">
            Helpful, with clear limits
          </h2>

          <p className="mx-auto mt-4 max-w-3xl leading-7 text-slate-600">
            DermaWise can only recognize patterns from the categories it was trained to support. It is not clinically validated and should not replace professional medical care.
          </p>

        </div>
      </section>

      {/* FOOTER */}

      <footer className="border-t border-slate-200 bg-slate-950 px-5 py-8 text-center">

        <p className="font-bold text-white">
          DermaWise AI
        </p>

        <p className="mt-2 text-sm text-slate-400">
          AI-assisted dermatology research
          prototype
        </p>

      </footer>

    </main>
  );
}