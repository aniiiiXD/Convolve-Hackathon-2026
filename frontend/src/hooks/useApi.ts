"use client";

import { useState, useCallback } from "react";

interface UseApiState<T> {
  data: T | null;
  error: Error | null;
  isLoading: boolean;
}

interface UseApiReturn<T, P extends unknown[]> extends UseApiState<T> {
  execute: (...args: P) => Promise<T | null>;
  reset: () => void;
}

export function useApi<T, P extends unknown[] = []>(
  apiFunc: (...args: P) => Promise<T>
): UseApiReturn<T, P> {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    error: null,
    isLoading: false,
  });

  const execute = useCallback(
    async (...args: P): Promise<T | null> => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        const data = await apiFunc(...args);
        setState({ data, error: null, isLoading: false });
        return data;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        setState({ data: null, error: err, isLoading: false });
        return null;
      }
    },
    [apiFunc]
  );

  const reset = useCallback(() => {
    setState({ data: null, error: null, isLoading: false });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

// Specific hooks for common API calls
import api from "@/lib/api";
import type {
  SearchParams,
  DiagnosisParams,
  InsightsParams,
  EvidenceGraphParams,
} from "@/lib/api";

export function useSearch() {
  return useApi((params: SearchParams) => api.advancedSearch(params));
}

export function useDiagnosis() {
  return useApi((params: DiagnosisParams) => api.generateDifferentialDiagnosis(params));
}

export function useInsights() {
  return useApi((params: InsightsParams) => api.generateInsights(params));
}

export function useAlerts(patientId?: string) {
  return useApi(() => api.getAlerts(patientId));
}

export function useEvidenceGraph() {
  return useApi((params: EvidenceGraphParams) => api.buildEvidenceGraph(params));
}

export function usePatients() {
  return useApi(() => api.getPatients());
}

export function usePatient(patientId: string) {
  return useApi(() => api.getPatient(patientId));
}

export function useComprehensiveAnalysis() {
  return useApi((patientId: string, chiefComplaint: string) =>
    api.getComprehensiveAnalysis(patientId, chiefComplaint)
  );
}
