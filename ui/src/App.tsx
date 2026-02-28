import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  AutoComplete,
  Badge,
  Button,
  Card,
  DatePicker,
  Descriptions,
  Divider,
  Drawer,
  Form,
  Input,
  InputNumber,
  Layout,
  List,
  message,
  Modal,
  Pagination,
  Segmented,
  Select,
  Switch,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd';
import type { ColumnsType, TablePaginationConfig } from 'antd/es/table';
import type { SorterResult } from 'antd/es/table/interface';
import type { Dayjs } from 'dayjs';
import { CopyOutlined, DownloadOutlined, ReloadOutlined, EyeOutlined } from '@ant-design/icons';
import {
  ArtifactLink,
  ArchiveBinDetail,
  ArchiveSummaryResponse,
  BuildDetailResponse,
  BuildInventoryStatsResponse,
  BuildStatus,
  BuildSummary,
  ConstraintStatus,
  GenerationRequest,
  GenerationRunSummary,
  ImportBuildMetadata,
  ImportBuildPayload,
  ImportBuildResponse,
  ModelOpsArtifactState,
  MLLoopStatusResponse,
  ModelOpsModelRecord,
  ModelOpsStatusResponse,
  ScenarioMetricSummary,
  evaluateBuild,
  fetchArchive,
  fetchArchiveFrontier,
  fetchBuildInventoryStats,
  fetchCodeExport,
  fetchMLLoopStatus,
  startMLLoop,
  stopMLLoop,
  fetchModelOpsStatus,
  fetchXmlExport,
  getBuildDetail,
  generateRun,
  importBuild,
  listBuilds,
} from './api/client';
import './index.css';

type ImportFormValues = {
  payload: string;
  metadata: {
    ruleset_id: string;
    profile_id: string;
    class: string;
    ascendancy: string;
    main_skill: string;
    damage_type: string;
    defence_type: string;
    complexity_bucket: string;
    tags: string;
  };
};

const metadataDefaults: ImportFormValues['metadata'] = {
  ruleset_id: 'poe-standard',
  profile_id: '',
  class: 'unknown',
  ascendancy: 'unknown',
  main_skill: 'unknown',
  damage_type: 'unknown',
  defence_type: 'unknown',
  complexity_bucket: 'medium',
  tags: '',
};

type GenerationFormValues = GenerationRequest;

const generationDefaults: GenerationRequest = {
  count: 3,
  seed_start: 1,
  ruleset_id: 'pob:local|scenarios:pinnacle@v0|prices:local',
  profile_id: 'pinnacle',
};

const statusColors: Record<BuildStatus, string> = {
  imported: 'default',
  queued: 'gold',
  evaluated: 'green',
  failed: 'red',
};

const statusOptions = [
  { label: 'Imported', value: 'imported' },
  { label: 'Queued', value: 'queued' },
  { label: 'Evaluated', value: 'evaluated' },
  { label: 'Failed', value: 'failed' },
];

const constraintStatusColors: Record<ConstraintStatus, string> = {
  pass: 'green',
  fail: 'volcano',
  unknown: 'gold',
};

const constraintStatusOptions = [
  { label: 'Pass', value: 'pass' },
  { label: 'Fail', value: 'fail' },
  { label: 'Unknown', value: 'unknown' },
];

const gatePassOptions = [
  { label: 'Any', value: 'any' },
  { label: 'Gate Passed', value: 'pass' },
  { label: 'Gate Failed', value: 'fail' },
];

const predictionModeOptions = [
  { label: 'Include predicted', value: 'include_predicted' },
  { label: 'Verified only', value: 'verified_only' },
];

const profileMetadata: Record<string, { description: string; gateSummary: string }> = {
  mapping: {
    description: 'Mapping builds tuned for fast clears, mobility, and sustain across atlas tiers.',
    gateSummary: 'Fast-clear thresholds for baseline survivability and sustain.',
  },
  pinnacle: {
    description: 'Pinnacle builds aim for boss gating and defenses outlined in EP-V7-01.',
    gateSummary: 'Boss thresholds for max-hit durability, resist stability, and DPS checks.',
  },
  uber: {
    description: 'Uber builds stress the highest-tier boss gates and emergency survivability.',
    gateSummary: 'Stricter pinnacle thresholds tuned for extreme burst and attrition windows.',
  },
  delve: {
    description: 'Delve profile focuses on deep-cave survivability, resistances, and pacing.',
    gateSummary: 'Tiered depth thresholds prioritizing max-hit, life pool, and resistance floor.',
  },
  support: {
    description: 'Support builds monitor aura, curse, and buff uptime in gate scenarios.',
    gateSummary:
      'Utility-focused checks for reservation efficiency, uptime, and party contribution.',
  },
};
const profileOptions = Object.keys(profileMetadata).map((value) => ({
  label: `${value} - ${profileMetadata[value].description}`,
  value,
}));
const normalizeProfileId = (value?: string) => {
  const normalized = (value ?? '').trim();
  if (!normalized) {
    return '';
  }
  const knownProfile = normalized.toLowerCase();
  if (knownProfile in profileMetadata) {
    return knownProfile;
  }
  return normalized;
};
const getProfileMetadata = (profileId?: string) =>
  profileId ? profileMetadata[normalizeProfileId(profileId)] : undefined;
const ProfileSummary = ({ profileId }: { profileId?: string }) => {
  const metadata = getProfileMetadata(profileId);
  if (!metadata) {
    return null;
  }
  return (
    <div className="profile-summary">
      <Typography.Text type="secondary">{metadata.description}</Typography.Text>
      <Typography.Text type="secondary">
        <strong>Gate summary:</strong> {metadata.gateSummary}
      </Typography.Text>
    </div>
  );
};

const formatDate = (value?: string) => {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
};

const formatPredictionValue = (value?: number | null) => (value != null ? value.toFixed(2) : '—');

const formatPredictionProbability = (value?: number | null) =>
  value != null ? `${(value * 100).toFixed(1)}%` : '—';

const scenarioColumns: ColumnsType<ScenarioMetricSummary> = [
  {
    title: 'Scenario',
    dataIndex: 'scenario_id',
    key: 'scenario_id',
    render: (value) => (
      <Typography.Text code ellipsis>
        {value}
      </Typography.Text>
    ),
  },
  {
    title: 'Gate',
    dataIndex: 'gate_pass',
    key: 'gate_pass',
    render: (value) => <Tag color={value ? 'green' : 'volcano'}>{value ? 'pass' : 'fail'}</Tag>,
  },
  {
    title: 'Full DPS',
    dataIndex: 'full_dps',
    key: 'full_dps',
    render: (value) => value.toFixed(2),
  },
  {
    title: 'Max Hit',
    dataIndex: 'max_hit',
    key: 'max_hit',
    render: (value) => value.toFixed(2),
  },
  {
    title: 'Armour',
    dataIndex: 'armour',
    key: 'armour',
    responsive: ['md'],
  },
  {
    title: 'Evasion',
    dataIndex: 'evasion',
    key: 'evasion',
    responsive: ['lg'],
  },
  {
    title: 'Life',
    dataIndex: 'life',
    key: 'life',
  },
  {
    title: 'Mana',
    dataIndex: 'mana',
    key: 'mana',
    responsive: ['lg'],
  },
  {
    title: 'Utility',
    dataIndex: 'utility_score',
    key: 'utility_score',
  },
  {
    title: 'Evaluated',
    dataIndex: 'evaluated_at',
    key: 'evaluated_at',
    render: formatDate,
    responsive: ['lg'],
  },
];

type ScenarioDiffView = {
  leftScenario: string;
  rightScenario: string;
  rows: Array<{
    key: string;
    metric: string;
    leftValue: number;
    rightValue: number;
    delta: number;
  }>;
};

const App = () => {
  const [builds, setBuilds] = useState<BuildSummary[]>([]);
  const [loadingBuilds, setLoadingBuilds] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);
  const [profileFilter, setProfileFilter] = useState('');
  const [rulesetFilter, setRulesetFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<BuildStatus | ''>('');
  const [scenarioFilter, setScenarioFilter] = useState('');
  const [gatePassFilter, setGatePassFilter] = useState<'any' | 'pass' | 'fail'>('any');
  const [predictionModeFilter, setPredictionModeFilter] = useState<
    'include_predicted' | 'verified_only'
  >('include_predicted');
  const [maxCostChaosFilter, setMaxCostChaosFilter] = useState<number | null>(null);
  const [excludeUnknownCostFilter, setExcludeUnknownCostFilter] = useState(false);
  const [priceSnapshotFilter, setPriceSnapshotFilter] = useState('');
  const [costCalculatedRange, setCostCalculatedRange] = useState<[
    Dayjs | null,
    Dayjs | null,
  ]>([null, null]);
  const [constraintStatusFilter, setConstraintStatusFilter] = useState<ConstraintStatus | ''>('');
  const [constraintReasonCodeFilter, setConstraintReasonCodeFilter] = useState('');
  const [violatedConstraintFilter, setViolatedConstraintFilter] = useState('');
  const [constraintCheckedRange, setConstraintCheckedRange] = useState<[Dayjs | null, Dayjs | null]>([null, null]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [sortField, setSortField] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [selectedBuildId, setSelectedBuildId] = useState<string | null>(null);
  const [detail, setDetail] = useState<BuildDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [copyingCodeId, setCopyingCodeId] = useState<string | null>(null);
  const [downloadingXmlId, setDownloadingXmlId] = useState<string | null>(null);
  const [xmlViewerOpen, setXmlViewerOpen] = useState(false);
  const [xmlViewerContent, setXmlViewerContent] = useState<string | null>(null);
  const [xmlViewerLoading, setXmlViewerLoading] = useState(false);
  const [leftScenarioId, setLeftScenarioId] = useState<string>('');
  const [rightScenarioId, setRightScenarioId] = useState<string>('');

  const [importMode, setImportMode] = useState<'share' | 'xml'>('share');
  const [importForm] = Form.useForm<ImportFormValues>();
  const [importLoading, setImportLoading] = useState(false);
  const [importResult, setImportResult] = useState<ImportBuildResponse | null>(null);
  const [evaluatingIds, setEvaluatingIds] = useState<Set<string>>(new Set());
  const [generationForm] = Form.useForm<GenerationFormValues>();
  const [generationLoading, setGenerationLoading] = useState(false);
  const [generationResult, setGenerationResult] = useState<GenerationRunSummary | null>(null);
  const [archiveRunId, setArchiveRunId] = useState('');
  const [archiveLoading, setArchiveLoading] = useState(false);
  const [archiveError, setArchiveError] = useState<string | null>(null);
  const [archiveData, setArchiveData] = useState<ArchiveSummaryResponse | null>(null);
  const [selectedArchiveBin, setSelectedArchiveBin] = useState<ArchiveBinDetail | null>(null);
  const [archiveMode, setArchiveMode] = useState<'archive' | 'frontier'>('archive');
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [modelOpsStatus, setModelOpsStatus] = useState<ModelOpsStatusResponse | null>(null);
  const [modelOpsLoading, setModelOpsLoading] = useState(false);
  const [modelOpsError, setModelOpsError] = useState<string | null>(null);
  const [inventoryStats, setInventoryStats] = useState<BuildInventoryStatsResponse | null>(null);
  const [mlLoopStatus, setMLLoopStatus] = useState<MLLoopStatusResponse | null>(null);
  const [mlLoopLoading, setMLLoopLoading] = useState(false);
  const [mlLoopError, setMLLoopError] = useState<string | null>(null);
  const [mlLoopIdInput, setMLLoopIdInput] = useState('ml-loop-ui');
  const [mlLoopIterations, setMLLoopIterations] = useState(5);
  const [mlLoopEndless, setMLLoopEndless] = useState(true);
  const [mlLoopActionLoading, setMLLoopActionLoading] = useState(false);
  const [mlLoopActionFeedback, setMLLoopActionFeedback] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const gatePassDisabled = !scenarioFilter.trim();
  const ratioSortDisabled = !scenarioFilter.trim();
  const hasMediumSortNarrowingFilter = Boolean(
    rulesetFilter.trim() ||
      normalizeProfileId(profileFilter) ||
      scenarioFilter.trim() ||
      priceSnapshotFilter.trim(),
  );
  const mediumPrioritySortDisabled = !hasMediumSortNarrowingFilter || pageSize > 100;

  const refreshCatalog = useCallback(() => setRefreshKey((prev) => prev + 1), []);

  const fetchBuilds = useCallback(async () => {
    setLoadingBuilds(true);
    setBuildError(null);
    try {
      const gatePassValue =
        gatePassFilter === 'pass' ? true : gatePassFilter === 'fail' ? false : undefined;
      const data = await listBuilds({
        profileId: normalizeProfileId(profileFilter) || undefined,
        rulesetId: rulesetFilter || undefined,
        status: statusFilter || undefined,
        predictionMode: predictionModeFilter,
        scenarioId: scenarioFilter || undefined,
        gatePass: gatePassValue,
        maxCostChaos: maxCostChaosFilter ?? undefined,
        excludeUnknownCost: excludeUnknownCostFilter ? true : undefined,
        priceSnapshotId: priceSnapshotFilter || undefined,
        costCalculatedAfter: costCalculatedRange[0]?.toISOString(),
        costCalculatedBefore: costCalculatedRange[1]?.toISOString(),
        constraintStatus: constraintStatusFilter || undefined,
        constraintReasonCode: constraintReasonCodeFilter || undefined,
        violatedConstraint: violatedConstraintFilter || undefined,
        constraintCheckedAfter: constraintCheckedRange[0]?.toISOString(),
        constraintCheckedBefore: constraintCheckedRange[1]?.toISOString(),
        sortBy: sortField || undefined,
        sortDir: sortDirection || undefined,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      });
      setBuilds(data);
      setLastUpdated(new Date());
    } catch (err) {
      setBuildError((err as Error).message || 'Unable to load builds');
      setBuilds([]);
    } finally {
      setLoadingBuilds(false);
    }
  }, [
    gatePassFilter,
    page,
    pageSize,
    profileFilter,
    predictionModeFilter,
    rulesetFilter,
    scenarioFilter,
    sortDirection,
    sortField,
    statusFilter,
    maxCostChaosFilter,
    excludeUnknownCostFilter,
    priceSnapshotFilter,
    constraintStatusFilter,
    constraintReasonCodeFilter,
    violatedConstraintFilter,
    constraintCheckedRange,
    costCalculatedRange,
  ]);

  useEffect(() => {
    fetchBuilds();
  }, [fetchBuilds, refreshKey]);

  useEffect(() => {
    setPage(1);
  }, [
    profileFilter,
    rulesetFilter,
    statusFilter,
    predictionModeFilter,
    scenarioFilter,
    gatePassFilter,
    maxCostChaosFilter,
    excludeUnknownCostFilter,
    priceSnapshotFilter,
    constraintStatusFilter,
    constraintReasonCodeFilter,
    violatedConstraintFilter,
    constraintCheckedRange,
    costCalculatedRange,
  ]);

  useEffect(() => {
    if (gatePassDisabled && gatePassFilter !== 'any') {
      setGatePassFilter('any');
    }
  }, [gatePassDisabled, gatePassFilter]);

  useEffect(() => {
    if (
      (ratioSortDisabled || mediumPrioritySortDisabled) &&
      (sortField === 'dps_per_chaos' ||
        sortField === 'max_hit_per_chaos' ||
        sortField === 'total_cost_chaos')
    ) {
      setSortField(null);
      setSortDirection(null);
    }
  }, [ratioSortDisabled, mediumPrioritySortDisabled, sortField]);

  const loadDetail = useCallback(async (buildId: string) => {
    setDetailLoading(true);
    try {
      const data = await getBuildDetail(buildId);
      setDetail(data);
    } catch (err) {
      message.error((err as Error).message || 'Unable to load build detail');
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleTableChange = (
    _: TablePaginationConfig,
    __: Record<string, unknown>,
    sorter: SorterResult<BuildSummary> | SorterResult<BuildSummary>[],
  ) => {
    const activeSorter = Array.isArray(sorter) ? sorter[0] : sorter;
    if (activeSorter?.order) {
      setSortField(activeSorter.field as string);
      setSortDirection(activeSorter.order === 'ascend' ? 'asc' : 'desc');
    } else {
      setSortField(null);
      setSortDirection(null);
    }
  };

  const handleOpenDetail = useCallback(
    (buildId: string) => {
      setSelectedBuildId(buildId);
      setDrawerOpen(true);
      loadDetail(buildId);
    },
    [loadDetail],
  );

  const handleCloseDrawer = () => {
    setDrawerOpen(false);
    setDetail(null);
    setSelectedBuildId(null);
    setLeftScenarioId('');
    setRightScenarioId('');
  };

  const handleEvaluate = useCallback(
    async (buildId: string) => {
      setEvaluatingIds((prev) => new Set(prev).add(buildId));
      try {
        await evaluateBuild(buildId);
        message.success('Evaluation triggered');
        refreshCatalog();
        if (drawerOpen && selectedBuildId === buildId) {
          loadDetail(buildId);
        }
      } catch (err) {
        message.error((err as Error).message || 'Unable to evaluate build');
      } finally {
        setEvaluatingIds((prev) => {
          const next = new Set(prev);
          next.delete(buildId);
          return next;
        });
      }
    },
    [drawerOpen, loadDetail, refreshCatalog, selectedBuildId],
  );

  const copyTextToClipboard = useCallback(async (value: string) => {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const textArea = document.createElement('textarea');
    textArea.value = value;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const success = document.execCommand('copy');
    textArea.remove();
    if (!success) {
      throw new Error('Clipboard is unavailable in this browser context');
    }
  }, []);

  const handleCopyCode = async (buildId: string) => {
    setCopyingCodeId(buildId);
    try {
      const code = await fetchCodeExport(buildId);
      await copyTextToClipboard(code);
      message.success('PoB import code copied to clipboard');
    } catch (err) {
      message.error((err as Error).message || 'Unable to copy code');
    } finally {
      setCopyingCodeId(null);
    }
  };

  const handleDownloadXml = async (buildId: string) => {
    setDownloadingXmlId(buildId);
    try {
      const xml = await fetchXmlExport(buildId);
      const blob = new Blob([xml], { type: 'application/xml' });
      const objectUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = `${buildId}.xml`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(objectUrl);
      message.success('XML downloaded');
    } catch (err) {
      message.error((err as Error).message || 'Unable to download XML');
    } finally {
      setDownloadingXmlId(null);
    }
  };

  const handleViewXml = async (buildId: string) => {
    setXmlViewerLoading(true);
    try {
      const xml = await fetchXmlExport(buildId);
      setXmlViewerContent(xml);
      setXmlViewerOpen(true);
    } catch (err) {
      message.error((err as Error).message || 'Unable to fetch XML');
    } finally {
      setXmlViewerLoading(false);
    }
  };

  const handleCloseXmlViewer = () => {
    setXmlViewerOpen(false);
    setXmlViewerContent(null);
  };

  const handleImport = async (values: ImportFormValues) => {
    setImportLoading(true);
    try {
      const tagsValue = values.metadata.tags ?? '';
      const tags = String(tagsValue)
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
      const metadata: ImportBuildMetadata = {
        ruleset_id: values.metadata.ruleset_id,
        profile_id: normalizeProfileId(values.metadata.profile_id),
        class: values.metadata.class,
        ascendancy: values.metadata.ascendancy,
        main_skill: values.metadata.main_skill,
        damage_type: values.metadata.damage_type,
        defence_type: values.metadata.defence_type,
        complexity_bucket: values.metadata.complexity_bucket,
        tags,
      };
      const payload: ImportBuildPayload = {
        metadata,
        ...(importMode === 'share' ? { share_code: values.payload } : { xml: values.payload }),
      };
      const response = await importBuild(payload);
      setImportResult(response);
      message.success(`Imported build ${response.build_id}`);
      importForm.resetFields(['payload']);
      refreshCatalog();
    } catch (err) {
      message.error((err as Error).message || 'Import failed');
    } finally {
      setImportLoading(false);
    }
  };

  const handleGeneration = async (values: GenerationFormValues) => {
    setGenerationLoading(true);
    setGenerationError(null);
    try {
      const summary = await generateRun({
        ...values,
        profile_id: normalizeProfileId(values.profile_id),
      });
      setGenerationResult(summary);
      refreshCatalog();
      refreshModelOpsStatus();
      refreshInventoryStats();
      refreshMLLoopStatus();
      message.success(`Generation ${summary.run_id} finished`);
    } catch (err) {
      const next = (err as Error).message || 'Generation failed';
      setGenerationError(next);
      message.error(next);
    } finally {
      setGenerationLoading(false);
    }
  };

  const handleArchiveLoad = useCallback(async () => {
    const runId = archiveRunId.trim();
    if (!runId) {
      setArchiveError('Enter a run ID to load the archive');
      return;
    }
    setArchiveLoading(true);
    setArchiveError(null);
    try {
      const payload = await fetchArchive(runId);
      setArchiveData(payload);
      setSelectedArchiveBin(null);
      setArchiveMode('archive');
    } catch (err) {
      const next = (err as Error).message || 'Unable to load archive';
      setArchiveError(next);
      setArchiveData(null);
      setSelectedArchiveBin(null);
      setArchiveMode('archive');
    } finally {
      setArchiveLoading(false);
    }
  }, [archiveRunId]);

  const handleArchiveFrontierLoad = useCallback(async () => {
    const runId = archiveRunId.trim();
    if (!runId) {
      setArchiveError('Enter a run ID to load the archive');
      return;
    }
    setArchiveLoading(true);
    setArchiveError(null);
    try {
      const payload = await fetchArchiveFrontier(runId);
      setArchiveData(payload);
      setSelectedArchiveBin(null);
      setArchiveMode('frontier');
    } catch (err) {
      const next = (err as Error).message || 'Unable to load archive';
      setArchiveError(next);
      setArchiveData(null);
      setSelectedArchiveBin(null);
      setArchiveMode('archive');
    } finally {
      setArchiveLoading(false);
    }
  }, [archiveRunId]);

  const columns: ColumnsType<BuildSummary> = [
    {
      title: 'Build ID',
      dataIndex: 'build_id',
      key: 'build_id',
      render: (value) => (
        <Typography.Text copyable ellipsis={{ tooltip: value }}>
          {value.slice(0, 8)}
        </Typography.Text>
      ),
    },
    {
      title: 'Profile',
      dataIndex: 'profile_id',
      key: 'profile_id',
      responsive: ['sm'],
    },
    {
      title: 'Ruleset',
      dataIndex: 'ruleset_id',
      key: 'ruleset_id',
      responsive: ['sm'],
    },
    {
      title: 'Class',
      dataIndex: 'class_',
      key: 'class_',
    },
    {
      title: 'Ascendancy',
      dataIndex: 'ascendancy',
      key: 'ascendancy',
      responsive: ['md'],
    },
    {
      title: 'Main Skill',
      dataIndex: 'main_skill',
      key: 'main_skill',
      responsive: ['md'],
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (value: BuildStatus) => <Tag color={statusColors[value]}>{value}</Tag>,
    },
    {
      title: 'Prediction',
      key: 'prediction',
      render: (_, record) => {
        const prediction = record.prediction;
        if (!prediction) {
          return <Typography.Text type="secondary">—</Typography.Text>;
        }
        const state = record.status === 'evaluated' ? 'Verified' : 'Predicted';
        return (
          <Space direction="vertical" size={0} className="prediction-metrics">
            <Tag color={state === 'Verified' ? 'green' : 'cyan'}>{state}</Tag>
            <Space size={4} wrap>
              {prediction.predicted_full_dps != null && (
                <Typography.Text
                  code
                >{`DPS ${prediction.predicted_full_dps.toFixed(0)}`}</Typography.Text>
              )}
              {prediction.predicted_max_hit != null && (
                <Typography.Text
                  code
                >{`Max ${prediction.predicted_max_hit.toFixed(0)}`}</Typography.Text>
              )}
            </Space>
            {prediction.pass_probability != null && (
              <Typography.Text type="secondary">
                {formatPredictionProbability(prediction.pass_probability)} pass
              </Typography.Text>
            )}
          </Space>
        );
      },
      responsive: ['lg'],
    },
    {
      title: 'Constraints',
      key: 'constraints',
      render: (_, record) => {
        const status = record.constraint_status as ConstraintStatus | undefined;
        const reason = record.constraint_reason_code;
        const violations = record.violated_constraints ?? [];
        if (!status && !reason && !violations.length) {
          return <Typography.Text type="secondary">—</Typography.Text>;
        }
        return (
          <Space direction="vertical" size={0} style={{ minWidth: 140 }}>
            {status ? (
              <Tag color={constraintStatusColors[status]}>{status}</Tag>
            ) : (
              <Typography.Text type="secondary">Status unknown</Typography.Text>
            )}
            {reason && (
              <Typography.Text type="secondary">{reason}</Typography.Text>
            )}
            {violations.length ? (
              <Typography.Text type="secondary">
                {violations.length} violation
                {violations.length > 1 ? 's' : ''}
              </Typography.Text>
            ) : null}
          </Space>
        );
      },
      responsive: ['lg'],
    },
    {
      title: 'Total Cost',
      dataIndex: 'total_cost_chaos',
      key: 'total_cost_chaos',
      sorter: !mediumPrioritySortDisabled,
      sortOrder:
        !mediumPrioritySortDisabled && sortField === 'total_cost_chaos'
          ? sortDirection === 'asc'
            ? 'ascend'
            : 'descend'
          : undefined,
      render: (value) => (value != null ? value.toFixed(2) : '—'),
      responsive: ['md'],
    },
    {
      title: 'DPS/Chaos',
      dataIndex: 'dps_per_chaos',
      key: 'dps_per_chaos',
      sorter: !ratioSortDisabled && !mediumPrioritySortDisabled,
      sortOrder:
        !ratioSortDisabled && !mediumPrioritySortDisabled && sortField === 'dps_per_chaos'
          ? sortDirection === 'asc'
            ? 'ascend'
            : 'descend'
          : undefined,
      render: (value) => (value != null ? value.toFixed(2) : '—'),
      responsive: ['lg'],
    },
    {
      title: 'MaxHit/Chaos',
      dataIndex: 'max_hit_per_chaos',
      key: 'max_hit_per_chaos',
      sorter: !ratioSortDisabled && !mediumPrioritySortDisabled,
      sortOrder:
        !ratioSortDisabled && !mediumPrioritySortDisabled && sortField === 'max_hit_per_chaos'
          ? sortDirection === 'asc'
            ? 'ascend'
            : 'descend'
          : undefined,
      render: (value) => (value != null ? value.toFixed(2) : '—'),
      responsive: ['lg'],
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      sorter: true,
      sortOrder:
        sortField === 'created_at' ? (sortDirection === 'asc' ? 'ascend' : 'descend') : undefined,
      render: formatDate,
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            onClick={(event) => {
              event.stopPropagation();
              handleOpenDetail(record.build_id);
            }}
          >
            Details
          </Button>
          <Button
            type="primary"
            size="small"
            onClick={(event) => {
              event.stopPropagation();
              handleEvaluate(record.build_id);
            }}
            loading={evaluatingIds.has(record.build_id)}
          >
            Evaluate
          </Button>
        </Space>
      ),
    },
  ];

  const archiveColumns = useMemo<ColumnsType<ArchiveBinDetail>>(
    () => [
      {
        title: 'Bin',
        dataIndex: 'bin_key',
        key: 'bin_key',
      },
      {
        title: 'Build ID',
        dataIndex: 'build_id',
        key: 'build_id',
        render: (value: string) => (
          <Typography.Text copyable ellipsis={{ tooltip: value }}>
            {value.slice(0, 8)}
          </Typography.Text>
        ),
      },
      {
        title: 'Score',
        dataIndex: 'score',
        key: 'score',
        render: (value: number) => value.toFixed(2),
      },
      {
        title: 'Descriptor',
        dataIndex: 'descriptor',
        key: 'descriptor',
        render: (descriptor: Record<string, number>) => (
          <Space wrap size="small">
            {Object.entries(descriptor).map(([axis, value]) => (
              <Tag key={axis}>{`${axis}: ${value.toFixed(2)}`}</Tag>
            ))}
          </Space>
        ),
      },
    ],
    [],
  );

  const gateFailures = useMemo(() => {
    const reasons = detail?.scenario_metrics.flatMap((item) => item.gate_fail_reasons ?? []) ?? [];
    return Array.from(new Set(reasons));
  }, [detail]);

  const warnings = useMemo(() => {
    const entries = detail?.scenario_metrics.flatMap((item) => item.pob_warnings ?? []) ?? [];
    return Array.from(new Set(entries));
  }, [detail]);

  const normalizedArchiveArtifactLinks = useMemo<ArtifactLink[]>(() => {
    if (!selectedArchiveBin || !selectedArchiveBin.artifact_links) {
      return [];
    }
    const links = selectedArchiveBin.artifact_links;
    if (Array.isArray(links)) {
      return links;
    }
    const recordLinks = links as Record<string, string>;
    return Object.entries(recordLinks).map(([label, url]) => ({ label, url }));
  }, [selectedArchiveBin]);

  const scenarioIds = useMemo(
    () => Array.from(new Set(detail?.scenario_metrics.map((item) => item.scenario_id) ?? [])),
    [detail],
  );

  const costDetails = detail?.costs ?? null;
  const compositionDetails = detail?.build_details ?? null;
  const compositionIdentity =
    compositionDetails && typeof compositionDetails.identity === 'object'
      ? (compositionDetails.identity as Record<string, unknown>)
      : null;
  const compositionSource =
    compositionDetails && typeof compositionDetails.source === 'string'
      ? compositionDetails.source
      : null;
  const compositionItems = useMemo(() => {
    if (!compositionDetails || typeof compositionDetails.items !== 'object') {
      return [] as Array<Record<string, unknown>>;
    }
    const payload = compositionDetails.items as Record<string, unknown>;
    if (Array.isArray(payload.items)) {
      return payload.items.filter((item): item is Record<string, unknown> =>
        Boolean(item && typeof item === 'object'),
      );
    }
    if (Array.isArray(payload.slot_templates)) {
      return payload.slot_templates.filter((item): item is Record<string, unknown> =>
        Boolean(item && typeof item === 'object'),
      );
    }
    return [] as Array<Record<string, unknown>>;
  }, [compositionDetails]);
  const compositionGemGroups = useMemo(() => {
    if (!compositionDetails || typeof compositionDetails.gems !== 'object') {
      return [] as Array<Record<string, unknown>>;
    }
    const payload = compositionDetails.gems as Record<string, unknown>;
    if (!Array.isArray(payload.groups)) {
      return [] as Array<Record<string, unknown>>;
    }
    return payload.groups.filter((item): item is Record<string, unknown> =>
      Boolean(item && typeof item === 'object'),
    );
  }, [compositionDetails]);
  const compositionPassiveNodes = useMemo(() => {
    if (!compositionDetails || typeof compositionDetails.passives !== 'object') {
      return [] as string[];
    }
    const payload = compositionDetails.passives as Record<string, unknown>;
    if (!Array.isArray(payload.node_ids)) {
      return [] as string[];
    }
    return payload.node_ids
      .map((item) => (typeof item === 'string' || typeof item === 'number' ? String(item) : null))
      .filter((item): item is string => item !== null);
  }, [compositionDetails]);

  const detailPrediction = detail?.prediction ?? null;
  const detailPredictionState = detailPrediction
    ? detailPrediction.verified_full_dps != null || detailPrediction.verified_max_hit != null
      ? 'Verified'
      : 'Predicted'
    : null;
  const detailPredictionColor =
    detailPredictionState === 'Verified'
      ? 'green'
      : detailPredictionState === 'Predicted'
        ? 'cyan'
        : undefined;

  const detailConstraintEvaluation = detail?.constraints?.evaluation ?? null;
  const detailConstraintStatus = (
    detailConstraintEvaluation?.status ?? detail?.build.constraint_status
  ) as ConstraintStatus | undefined;
  const detailConstraintReason =
    detailConstraintEvaluation?.reason_code ?? detail?.build.constraint_reason_code ?? null;
  const detailConstraintCheckedAt =
    detailConstraintEvaluation?.checked_at ?? detail?.build.constraint_checked_at ?? null;
  const detailViolatedConstraints =
    detailConstraintEvaluation?.violated_constraints ?? detail?.build.violated_constraints ?? [];
  const hasConstraintMetadata = Boolean(
    detailConstraintStatus ||
      detailConstraintReason ||
      detailConstraintCheckedAt ||
      detailViolatedConstraints.length,
  );

  useEffect(() => {
    if (scenarioIds.length >= 2) {
      setLeftScenarioId(scenarioIds[0]);
      setRightScenarioId(scenarioIds[1]);
      return;
    }
    setLeftScenarioId('');
    setRightScenarioId('');
  }, [scenarioIds]);

  const refreshModelOpsStatus = useCallback(async () => {
    setModelOpsLoading(true);
    setModelOpsError(null);
    try {
      const payload = await fetchModelOpsStatus();
      setModelOpsStatus(payload);
    } catch (err) {
      setModelOpsError((err as Error).message || 'Unable to load model status');
    } finally {
      setModelOpsLoading(false);
    }
  }, []);

  const refreshInventoryStats = useCallback(async () => {
    try {
      const payload = await fetchBuildInventoryStats();
      setInventoryStats(payload);
    } catch {
      setInventoryStats(null);
    }
  }, []);

  const refreshMLLoopStatus = useCallback(async () => {
    setMLLoopLoading(true);
    setMLLoopError(null);
    try {
      const payload = await fetchMLLoopStatus();
      setMLLoopStatus(payload);
    } catch (err) {
      setMLLoopError((err as Error).message || 'Unable to load ML loop status');
      setMLLoopStatus(null);
    } finally {
      setMLLoopLoading(false);
    }
  }, []);

  const handleMLLoopStart = async () => {
    setMLLoopActionFeedback(null);
    setMLLoopActionLoading(true);
    try {
      const trimmedLoopId = mlLoopIdInput.trim() || 'ml-loop-ui';
      const payload = {
        loop_id: trimmedLoopId,
        profile_id: 'pinnacle',
        count: 3,
        seed_start: 1,
        surrogate_backend: 'cpu' as const,
        endless: mlLoopEndless,
        ...(mlLoopEndless ? {} : { iterations: mlLoopIterations }),
      };
      const response = await startMLLoop(payload);
      setMLLoopActionFeedback({
        type: 'success',
        message: response.message || `Loop ${response.loop_id} started`,
      });
      refreshMLLoopStatus();
    } catch (error) {
      setMLLoopActionFeedback({
        type: 'error',
        message: (error as Error).message || 'Unable to start ML loop',
      });
    } finally {
      setMLLoopActionLoading(false);
    }
  };

  const handleMLLoopStop = async () => {
    setMLLoopActionFeedback(null);
    setMLLoopActionLoading(true);
    try {
      const payload = mlLoopIdInput.trim() ? { loop_id: mlLoopIdInput.trim() } : {};
      const response = await stopMLLoop(payload);
      setMLLoopActionFeedback({
        type: 'success',
        message: response.message || `Stop requested for ${response.loop_id}`,
      });
      refreshMLLoopStatus();
    } catch (error) {
      setMLLoopActionFeedback({
        type: 'error',
        message: (error as Error).message || 'Unable to stop ML loop',
      });
    } finally {
      setMLLoopActionLoading(false);
    }
  };


  useEffect(() => {
    refreshModelOpsStatus();
    refreshInventoryStats();
    refreshMLLoopStatus();
  }, [refreshModelOpsStatus, refreshInventoryStats, refreshMLLoopStatus, refreshKey]);

  useEffect(() => {
    const interval = setInterval(() => {
      refreshModelOpsStatus();
      refreshMLLoopStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [refreshModelOpsStatus, refreshMLLoopStatus]);

  const scenarioDiff = useMemo<ScenarioDiffView | null>(() => {
    if (!detail || detail.scenario_metrics.length < 2 || !leftScenarioId || !rightScenarioId) {
      return null;
    }
    if (leftScenarioId === rightScenarioId) {
      return null;
    }
    const left = detail.scenario_metrics.find((item) => item.scenario_id === leftScenarioId);
    const right = detail.scenario_metrics.find((item) => item.scenario_id === rightScenarioId);
    if (!left || !right) {
      return null;
    }
    const metricRows: Array<{ key: string; label: string; left: number; right: number }> = [
      { key: 'full_dps', label: 'Full DPS', left: left.full_dps, right: right.full_dps },
      { key: 'max_hit', label: 'Max Hit', left: left.max_hit, right: right.max_hit },
      { key: 'armour', label: 'Armour', left: left.armour, right: right.armour },
      { key: 'evasion', label: 'Evasion', left: left.evasion, right: right.evasion },
      { key: 'life', label: 'Life', left: left.life, right: right.life },
      { key: 'mana', label: 'Mana', left: left.mana, right: right.mana },
      {
        key: 'utility_score',
        label: 'Utility',
        left: left.utility_score,
        right: right.utility_score,
      },
    ];
    return {
      leftScenario: left.scenario_id,
      rightScenario: right.scenario_id,
      rows: metricRows
        .map((row) => ({
          key: row.key,
          metric: row.label,
          leftValue: row.left,
          rightValue: row.right,
          delta: row.right - row.left,
        }))
        .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)),
    };
  }, [detail, leftScenarioId, rightScenarioId]);

  const inventoryStatusCounts = inventoryStats?.status_counts ?? {};
  const mlImprovement =
    mlLoopStatus?.last_improvement && typeof mlLoopStatus.last_improvement === 'object'
      ? (mlLoopStatus.last_improvement as Record<string, unknown>)
      : null;
  const mlImproved = typeof mlImprovement?.improved === 'boolean' ? mlImprovement.improved : null;
  const mlPassProbabilityDelta =
    typeof mlImprovement?.pass_probability_mean_delta === 'number'
      ? mlImprovement.pass_probability_mean_delta
      : null;
  const mlMaeDeltas =
    mlImprovement && typeof mlImprovement.metric_mae_deltas === 'object'
      ? (mlImprovement.metric_mae_deltas as Record<string, number>)
      : {};
  const mlBestIteration =
    typeof mlLoopStatus?.best_iteration === 'number' ? mlLoopStatus.best_iteration : null;
  const mlBestPassProbability =
    typeof mlLoopStatus?.best_pass_probability_mean === 'number'
      ? mlLoopStatus.best_pass_probability_mean
      : null;
  const passTrend = Array.isArray(mlLoopStatus?.pass_probability_trend)
    ? mlLoopStatus.pass_probability_trend
    : [];
  const gateTrend = Array.isArray(mlLoopStatus?.gate_pass_rate_trend)
    ? mlLoopStatus.gate_pass_rate_trend
    : [];
  const diversityTrend = Array.isArray(mlLoopStatus?.diversity_trend)
    ? mlLoopStatus.diversity_trend
    : [];
  const latestPassTrend =
    passTrend.length > 0 && typeof passTrend[passTrend.length - 1]?.mean === 'number'
      ? passTrend[passTrend.length - 1].mean
      : null;
  const latestGateTrend =
    gateTrend.length > 0 && typeof gateTrend[gateTrend.length - 1]?.mean === 'number'
      ? gateTrend[gateTrend.length - 1].mean
      : null;
  const latestDiversityTrend =
    diversityTrend.length > 0 && typeof diversityTrend[diversityTrend.length - 1]?.unique_main_skills === 'number'
      ? diversityTrend[diversityTrend.length - 1].unique_main_skills
      : null;

  const renderModelRecord = (record?: ModelOpsModelRecord | null) => {
    if (!record) {
      return <Typography.Text type="secondary">Not available</Typography.Text>;
    }
    return (
      <Space direction="vertical" size={0}>
        <Typography.Text strong>{record.model_id ?? 'Unnamed model'}</Typography.Text>
        {record.path && (
          <Typography.Text copyable ellipsis={{ tooltip: record.path }}>
            {record.path}
          </Typography.Text>
        )}
        <Typography.Text type="secondary">
          Trained: {formatDate(record.trained_at_utc ?? undefined)}
        </Typography.Text>
      </Space>
    );
  };

  const renderArtifactState = (state: ModelOpsArtifactState) => {
    const hasMetadata = state.metadata && Object.keys(state.metadata).length > 0;
    return (
      <Space direction="vertical" size={0}>
        <Space size={4} wrap>
          <Tag color={state.state === 'available' ? 'green' : 'default'}>{state.state}</Tag>
          {state.path ? (
            <Typography.Text copyable ellipsis={{ tooltip: state.path }}>
              {state.path}
            </Typography.Text>
          ) : (
            <Typography.Text type="secondary">Path unavailable</Typography.Text>
          )}
        </Space>
        <Typography.Text type="secondary">
          {state.timestamp ? formatDate(state.timestamp) : 'Timestamp unavailable'}
        </Typography.Text>
        {hasMetadata ? (
          <Typography.Paragraph className="scenario-json" code>
            {JSON.stringify(state.metadata, null, 2)}
          </Typography.Paragraph>
        ) : (
          <Typography.Text type="secondary">Metadata unavailable</Typography.Text>
        )}
      </Space>
    );
  };

  return (
    <Layout className="app-shell">
      <Layout.Header className="app-shell__header">
        <div className="app-shell__header-inner">
          <div>
            <Typography.Title level={3} className="app-shell__title">
              BuildAtlas UI
            </Typography.Title>
            <Typography.Text className="app-shell__subtitle">
              Real-time catalog powered by the FastAPI backend.
            </Typography.Text>
          </div>
          <Space className="app-shell__header-actions">
            <Button icon={<ReloadOutlined />} onClick={refreshCatalog}>
              Refresh catalog
            </Button>
          </Space>
        </div>
      </Layout.Header>
      <Layout.Content className="app-shell__content">
        <Card variant="borderless" className="catalog-card">
          <div className="catalog-card__header">
            <div>
              <Typography.Title level={4}>Build Catalog</Typography.Title>
              <Typography.Text type="secondary">
                {lastUpdated
                  ? `Last refreshed ${lastUpdated.toLocaleTimeString()}`
                  : 'Loading catalog...'}
              </Typography.Text>
            </div>
          </div>
          <div className="catalog-card__inventory">
            <Space wrap>
              <Tag color="blue">Total: {inventoryStats?.total_builds ?? '...'}</Tag>
              <Tag color="green">Evaluated: {inventoryStatusCounts.evaluated ?? 0}</Tag>
              <Tag color="volcano">Failed: {inventoryStatusCounts.failed ?? 0}</Tag>
              <Tag color="default">Imported: {inventoryStatusCounts.imported ?? 0}</Tag>
              <Tag color="purple">Queued: {inventoryStatusCounts.queued ?? 0}</Tag>
              <Tag color="gold">Stale: {inventoryStats?.stale_builds ?? 0}</Tag>
            </Space>
          </div>
          <div className="catalog-card__filters">
            <div className="catalog-profile-selector">
              <AutoComplete
                placeholder="Profile ID"
                allowClear
                options={profileOptions}
                value={profileFilter || undefined}
                onChange={(value) => setProfileFilter(normalizeProfileId(value))}
                style={{ width: '100%' }}
              />
              <ProfileSummary profileId={profileFilter} />
            </div>
            <Input
              placeholder="Ruleset ID"
              allowClear
              value={rulesetFilter}
              onChange={(event) => setRulesetFilter(event.target.value)}
            />
            <Select
              allowClear
              placeholder="Status"
              value={statusFilter || undefined}
              onChange={(value) => setStatusFilter((value ?? '') as BuildStatus | '')}
              options={statusOptions}
              style={{ minWidth: 140 }}
            />
            <Select
              value={predictionModeFilter}
              onChange={(value) =>
                setPredictionModeFilter(value as 'include_predicted' | 'verified_only')
              }
              options={predictionModeOptions}
              style={{ minWidth: 180 }}
            />
            <Select
              value={gatePassFilter}
              onChange={(value) => setGatePassFilter(value)}
              options={gatePassOptions}
              disabled={gatePassDisabled}
              style={{ minWidth: 140 }}
            />
            <InputNumber
              placeholder="Max chaos cost"
              min={0}
              style={{ width: 150 }}
              value={maxCostChaosFilter ?? undefined}
              onChange={(value) => setMaxCostChaosFilter(value ?? null)}
            />
            <Space align="center" style={{ minWidth: 180 }}>
              <Switch
                checked={excludeUnknownCostFilter}
                onChange={(value) => setExcludeUnknownCostFilter(value)}
                checkedChildren="Exclude"
                unCheckedChildren="Include"
              />
              <Typography.Text>Exclude unknown cost</Typography.Text>
            </Space>
            <Input
              placeholder="Price snapshot ID"
              allowClear
              value={priceSnapshotFilter}
              onChange={(event) => setPriceSnapshotFilter(event.target.value)}
            />
            <DatePicker.RangePicker
              value={costCalculatedRange}
              onChange={(value) => setCostCalculatedRange((value as [Dayjs | null, Dayjs | null]) ?? [null, null])}
              showTime
            />
            <Select
              allowClear
              placeholder="Constraint status"
              value={constraintStatusFilter || undefined}
              onChange={(value) => setConstraintStatusFilter((value ?? '') as ConstraintStatus | '')}
              options={constraintStatusOptions}
              style={{ minWidth: 160 }}
            />
            <Input
              placeholder="Constraint reason code"
              allowClear
              value={constraintReasonCodeFilter}
              onChange={(event) => setConstraintReasonCodeFilter(event.target.value)}
            />
            <Input
              placeholder="Violated constraint"
              allowClear
              value={violatedConstraintFilter}
              onChange={(event) => setViolatedConstraintFilter(event.target.value)}
            />
            <DatePicker.RangePicker
              value={constraintCheckedRange}
              placeholder={['Checked after', 'Checked before']}
              onChange={(value) =>
                setConstraintCheckedRange((value as [Dayjs | null, Dayjs | null]) ?? [null, null])
              }
              showTime
            />
            <Input
              placeholder="Scenario / gate"
              allowClear
              value={scenarioFilter}
              onChange={(event) => setScenarioFilter(event.target.value)}
            />
            <Button
              onClick={() => {
                setProfileFilter('');
                setRulesetFilter('');
                setStatusFilter('');
                setPredictionModeFilter('include_predicted');
                setScenarioFilter('');
                setGatePassFilter('any');
                setMaxCostChaosFilter(null);
                setExcludeUnknownCostFilter(false);
                setPriceSnapshotFilter('');
                setCostCalculatedRange([null, null]);
                setConstraintStatusFilter('');
                setConstraintReasonCodeFilter('');
                setViolatedConstraintFilter('');
                setConstraintCheckedRange([null, null]);
              }}
            >
              Reset filters
            </Button>
          </div>
          {gatePassDisabled && (
            <Typography.Text type="secondary">
              Provide a scenario filter to enable gate pass filtering.
            </Typography.Text>
          )}
          {mediumPrioritySortDisabled && (
            <Typography.Text type="secondary">
              Cost-oriented sorts require a narrowing filter (ruleset/profile/scenario/snapshot)
              and page size of 100 or less.
            </Typography.Text>
          )}
          {buildError && (
            <Alert message={buildError} type="error" showIcon className="catalog-card__alert" />
          )}
          <Table
            columns={columns}
            dataSource={builds}
            rowKey="build_id"
            loading={loadingBuilds}
            pagination={false}
            onChange={handleTableChange}
            onRow={(record) => ({ onClick: () => handleOpenDetail(record.build_id) })}
          />
          <div className="catalog-card__pagination">
            <Pagination
              current={page}
              pageSize={pageSize}
              total={
                inventoryStats?.total_builds ??
                (page - 1) * pageSize + builds.length + (builds.length === pageSize ? 1 : 0)
              }
              showSizeChanger
              pageSizeOptions={['5', '10', '20', '30']}
              onChange={(nextPage, nextSize) => {
                setPage(nextPage);
                setPageSize(nextSize);
              }}
              showTotal={(total) => `Showing ${builds.length} of ${total} builds`}
            />
          </div>
        </Card>
        <Card variant="borderless" className="import-card">
          <div className="import-card__header">
            <div>
              <Typography.Title level={4}>Import Build</Typography.Title>
              <Typography.Text type="secondary">
                Paste a PoB share code or the XML dump then enrich with metadata.
              </Typography.Text>
            </div>
          </div>
          <Form
            form={importForm}
            layout="vertical"
            initialValues={{ metadata: metadataDefaults }}
            onFinish={handleImport}
            className="import-form"
          >
            <Form.Item label="Import mode">
              <Segmented
                options={['share', 'xml'].map((value) => ({
                  label: value === 'share' ? 'Share code' : 'XML payload',
                  value,
                }))}
                value={importMode}
                onChange={(value) => setImportMode(value as 'share' | 'xml')}
              />
            </Form.Item>
            <Form.Item
              label={importMode === 'share' ? 'Share code' : 'Path of Building XML'}
              name="payload"
              rules={[{ required: true, message: 'Paste the payload for this import' }]}
            >
              <Input.TextArea
                rows={6}
                placeholder={
                  importMode === 'share'
                    ? 'paste the PoB share code (starts with ``build_name``)'
                    : 'paste the full PoB XML export'
                }
              />
            </Form.Item>
            <Divider />
            <Space className="metadata-grid" direction="vertical" size="small">
              <div className="metadata-row">
                <Form.Item label="Ruleset" name={['metadata', 'ruleset_id']}>
                  <Input />
                </Form.Item>
                <div className="profile-field">
                  <Form.Item label="Profile" name={['metadata', 'profile_id']}>
                    <AutoComplete
                      allowClear
                      options={profileOptions}
                      placeholder="Profile ID"
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                  <Form.Item
                    noStyle
                    shouldUpdate={(prev, current) =>
                      prev.metadata?.profile_id !== current.metadata?.profile_id
                    }
                  >
                    {({ getFieldValue }) => (
                      <ProfileSummary profileId={getFieldValue(['metadata', 'profile_id'])} />
                    )}
                  </Form.Item>
                </div>
              </div>
              <div className="metadata-row">
                <Form.Item label="Class" name={['metadata', 'class']}>
                  <Input />
                </Form.Item>
                <Form.Item label="Ascendancy" name={['metadata', 'ascendancy']}>
                  <Input />
                </Form.Item>
              </div>
              <div className="metadata-row">
                <Form.Item label="Main Skill" name={['metadata', 'main_skill']}>
                  <Input />
                </Form.Item>
                <Form.Item label="Tags" name={['metadata', 'tags']}>
                  <Input placeholder="comma separated" />
                </Form.Item>
              </div>
              <div className="metadata-row">
                <Form.Item label="Damage Type" name={['metadata', 'damage_type']}>
                  <Input />
                </Form.Item>
                <Form.Item label="Defence Type" name={['metadata', 'defence_type']}>
                  <Input />
                </Form.Item>
              </div>
              <div className="metadata-row">
                <Form.Item label="Complexity" name={['metadata', 'complexity_bucket']}>
                  <Input />
                </Form.Item>
              </div>
            </Space>
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={importLoading}>
                  Submit import
                </Button>
                <Button onClick={() => importForm.resetFields(['payload'])}>Clear payload</Button>
              </Space>
            </Form.Item>
          </Form>
          {importResult && (
            <Alert
              className="import-card__alert"
              message={`Imported ${importResult.build_id}`}
              description={
                <Space>
                  <Badge status="processing" />
                  <Typography.Text>status: {importResult.status}</Typography.Text>
                </Space>
              }
              type="success"
              showIcon
            />
          )}
        </Card>
        <Card variant="borderless" className="generation-card">
          <div className="generation-card__header">
            <div>
              <Typography.Title level={4} style={{ color: '#fff' }}>
                Generation Run
              </Typography.Title>
              <Typography.Text type="secondary" style={{ color: '#dbeafe' }}>
                Spin EP-V2-06 candidates and surface failure categories.
              </Typography.Text>
            </div>
          </div>
          <Form
            form={generationForm}
            layout="vertical"
            initialValues={generationDefaults}
            onFinish={handleGeneration}
            className="generation-card__form"
          >
            <div className="generation-card__form-grid">
              <Form.Item label="Count" name="count">
                <InputNumber min={1} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item label="Seed start" name="seed_start">
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
              <Form.Item label="Ruleset ID" name="ruleset_id">
                <Input />
              </Form.Item>
              <div className="profile-field">
                <Form.Item label="Profile ID" name="profile_id">
                  <AutoComplete
                    allowClear
                    options={profileOptions}
                    placeholder="Profile ID"
                    style={{ width: '100%' }}
                  />
                </Form.Item>
                <Form.Item
                  noStyle
                  shouldUpdate={(prev, current) => prev.profile_id !== current.profile_id}
                >
                  {({ getFieldValue }) => (
                    <ProfileSummary profileId={getFieldValue('profile_id')} />
                  )}
                </Form.Item>
              </div>
              <Form.Item label="Run ID (optional)" name="run_id">
                <Input />
              </Form.Item>
            </div>
            <Form.Item className="generation-card__submit">
              <Space wrap>
                <Button type="primary" htmlType="submit" loading={generationLoading}>
                  Launch generation
                </Button>
                <Button onClick={() => generationForm.resetFields()}>Reset</Button>
              </Space>
            </Form.Item>
            {generationError && <Alert message={generationError} type="error" showIcon />}
          </Form>
          {generationResult && (
            <div className="generation-card__result">
              <Descriptions
                column={2}
                size="small"
                bordered
                className="generation-card__summary"
              >
                <Descriptions.Item label="Run ID">{generationResult.run_id}</Descriptions.Item>
                <Descriptions.Item label="Status">{generationResult.status}</Descriptions.Item>
                <Descriptions.Item label="Requested">
                  {generationResult.generation.count}
                </Descriptions.Item>
                <Descriptions.Item label="Processed">
                  {generationResult.generation.processed}
                </Descriptions.Item>
              </Descriptions>
              <div className="generation-card__stats">
                <Badge count={generationResult.generation.processed} title="Generated builds" />
                <Badge count={generationResult.generation.count} title="Requested seeds" />
              </div>
              <div className="generation-card__failures">
                {Object.entries(generationResult.generation.precheck_failures).map(
                  ([key, count]) => (
                    <Tag key={key} color={count ? 'volcano' : 'default'}>
                      {key.replace(/_/g, ' ')}: {count}
                    </Tag>
                  ),
                )}
              </div>
              <div className="generation-card__evaluation">
                <Typography.Text strong>Evaluation</Typography.Text>
                <Space wrap>
                  <Tag color="blue">Attempted: {generationResult.evaluation.attempted}</Tag>
                  <Tag color="green">Successes: {generationResult.evaluation.successes}</Tag>
                  <Tag color="volcano">Failures: {generationResult.evaluation.failures}</Tag>
                  <Tag color="default">Errors: {generationResult.evaluation.errors}</Tag>
                </Space>
              </div>
            </div>
          )}
        </Card>
        <Card variant="borderless" className="model-ops-card">
          <div className="model-ops-card__header">
            <div>
              <Typography.Title level={4}>Model Ops</Typography.Title>
              <Typography.Text type="secondary">
                Monitor active models, checkpoints, and rollback readiness.
              </Typography.Text>
            </div>
            <Button
              icon={<ReloadOutlined />}
              loading={modelOpsLoading || mlLoopLoading}
              onClick={() => {
                refreshModelOpsStatus();
                refreshMLLoopStatus();
              }}
            >
              Refresh
            </Button>
          </div>
          <div className="model-ops-card__loop-controls">
            <Space wrap align="center">
              <Input
                placeholder="Loop ID"
                value={mlLoopIdInput}
                onChange={(event) => setMLLoopIdInput(event.target.value)}
                style={{ minWidth: 210 }}
              />
              <InputNumber
                min={1}
                value={mlLoopIterations}
                onChange={(value) => setMLLoopIterations(value ?? 1)}
                disabled={mlLoopEndless}
                style={{ width: 140 }}
                placeholder="Iterations"
              />
              <Space align="center">
                <Switch
                  checked={mlLoopEndless}
                  onChange={(value) => setMLLoopEndless(value)}
                  checkedChildren="Endless"
                  unCheckedChildren="Finite"
                />
                <Typography.Text type="secondary">Endless loop</Typography.Text>
              </Space>
              <Button
                type="primary"
                loading={mlLoopActionLoading}
                onClick={handleMLLoopStart}
              >
                Start loop
              </Button>
              <Button
                type="default"
                loading={mlLoopActionLoading}
                onClick={handleMLLoopStop}
              >
                Stop loop
              </Button>
            </Space>
            {mlLoopActionFeedback && (
              <Alert
                message={mlLoopActionFeedback.message}
                type={mlLoopActionFeedback.type}
                showIcon
                closable
                className="model-ops-card__loop-feedback"
              />
            )}
          </div>
          {modelOpsError && (
            <Alert
              className="model-ops-card__alert"
              message={modelOpsError}
              type="error"
              showIcon
            />
          )}
          {modelOpsLoading && !modelOpsStatus ? (
            <div className="model-ops-card__loading">
              <Spin />
            </div>
          ) : modelOpsStatus ? (
            <>
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="Generated">
                  {formatDate(modelOpsStatus.generated_at)}
                </Descriptions.Item>
                <Descriptions.Item label="Active model">
                  {renderModelRecord(modelOpsStatus.active_model)}
                </Descriptions.Item>
                <Descriptions.Item label="Last training">
                  {renderModelRecord(modelOpsStatus.last_training)}
                </Descriptions.Item>
                <Descriptions.Item label="Checkpoint">
                  {renderArtifactState(modelOpsStatus.checkpoint)}
                </Descriptions.Item>
                <Descriptions.Item label="Rollback">
                  {renderArtifactState(modelOpsStatus.rollback)}
                </Descriptions.Item>
              </Descriptions>
              <Divider />
              <Typography.Text strong>ML training trend</Typography.Text>
              {mlLoopError && (
                <Alert
                  className="model-ops-card__alert"
                  message={mlLoopError}
                  type="error"
                  showIcon
                  style={{ marginTop: 12 }}
                />
              )}
              {mlLoopLoading && !mlLoopStatus ? (
                <div className="model-ops-card__loading">
                  <Spin />
                </div>
              ) : mlLoopStatus?.loop_id ? (
                <div className="model-ops-card__trend">
                  <Space wrap>
                    <Tag color="processing">Loop: {mlLoopStatus.loop_id}</Tag>
                    <Tag>
                      Iteration: {mlLoopStatus.iteration ?? 0} / {mlLoopStatus.total_iterations ?? '?'}
                    </Tag>
                    {mlImproved !== null && (
                      <Tag color={mlImproved ? 'green' : 'volcano'}>
                        {mlImproved ? 'Improving' : 'Not improving yet'}
                      </Tag>
                    )}
                    {mlPassProbabilityDelta !== null && (
                      <Tag color={mlPassProbabilityDelta >= 0 ? 'green' : 'volcano'}>
                        Pass-prob delta: {mlPassProbabilityDelta >= 0 ? '+' : ''}
                        {mlPassProbabilityDelta.toFixed(4)}
                      </Tag>
                    )}
                    {mlBestPassProbability !== null && (
                      <Tag color="cyan">
                        Best pass-prob: {mlBestPassProbability.toFixed(4)}
                        {mlBestIteration !== null ? ` (iter ${mlBestIteration})` : ''}
                      </Tag>
                    )}
                    {latestPassTrend !== null && (
                      <Tag color="blue">Latest pass-prob: {latestPassTrend.toFixed(4)}</Tag>
                    )}
                    {latestGateTrend !== null && (
                      <Tag color={latestGateTrend >= 0.5 ? 'green' : 'volcano'}>
                        Gate pass rate: {(latestGateTrend * 100).toFixed(1)}%
                      </Tag>
                    )}
                    {latestDiversityTrend !== null && (
                      <Tag color="purple">Unique main skills: {Math.round(latestDiversityTrend)}</Tag>
                    )}
                  </Space>
                  {Object.keys(mlMaeDeltas).length ? (
                    <Space wrap style={{ marginTop: 8 }}>
                      {Object.entries(mlMaeDeltas).map(([metric, delta]) => (
                        <Tag key={metric} color={delta >= 0 ? 'green' : 'volcano'}>
                          {metric} delta: {delta >= 0 ? '+' : ''}
                          {delta.toFixed(4)}
                        </Tag>
                      ))}
                    </Space>
                  ) : (
                    <Typography.Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                      No MAE delta data available yet.
                    </Typography.Text>
                  )}
                  {mlLoopStatus.warnings.length ? (
                    <List
                      size="small"
                      dataSource={mlLoopStatus.warnings}
                      renderItem={(warning) => <List.Item>{warning}</List.Item>}
                      style={{ marginTop: 8 }}
                    />
                  ) : null}
                </div>
              ) : (
                <Typography.Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                  No ML loop status found yet.
                </Typography.Text>
              )}
              <Divider />
              <Typography.Text strong>Warnings</Typography.Text>
              {modelOpsStatus.warnings.length ? (
                <List
                  size="small"
                  dataSource={modelOpsStatus.warnings}
                  renderItem={(warning) => <List.Item>{warning}</List.Item>}
                />
              ) : (
                <Typography.Text type="secondary">No warnings recorded.</Typography.Text>
              )}
            </>
          ) : (
            <Typography.Text type="secondary">Model operations data is not available.</Typography.Text>
          )}
        </Card>
        <Card variant="borderless" className="archive-card">
          <div className="archive-card__header">
            <div>
              <Typography.Title level={4}>Archive Browser</Typography.Title>
              <Typography.Text type="secondary">
                Explore deterministic EP-V5 coverage for completed runs.
              </Typography.Text>
            </div>
            <Space wrap>
              <Input
                placeholder="Run ID"
                value={archiveRunId}
                onChange={(event) => setArchiveRunId(event.target.value)}
                onPressEnter={handleArchiveLoad}
                style={{ minWidth: 200 }}
              />
              <Button type="primary" loading={archiveLoading} onClick={handleArchiveLoad}>
                Load archive
              </Button>
              <Button
                type="default"
                loading={archiveLoading}
                onClick={handleArchiveFrontierLoad}
                disabled={!archiveRunId.trim()}
              >
                Load frontier
              </Button>
              <Button
                onClick={() => {
                  setArchiveRunId('');
                  setArchiveData(null);
                  setArchiveError(null);
                  setSelectedArchiveBin(null);
                  setArchiveMode('archive');
                }}
                disabled={!archiveData && !archiveError}
              >
                Clear
              </Button>
            </Space>
          </div>
          {archiveError && (
            <Alert message={archiveError} type="error" showIcon style={{ marginTop: 16 }} />
          )}
          {archiveData ? (
            <>
              <Descriptions column={3} size="small" bordered style={{ marginTop: 16 }}>
                <Descriptions.Item label="Bins filled">
                  {archiveData.metrics.bins_filled}
                </Descriptions.Item>
                <Descriptions.Item label="Total bins">
                  {archiveData.metrics.total_bins}
                </Descriptions.Item>
                <Descriptions.Item label="Coverage">
                  {(archiveData.metrics.coverage * 100).toFixed(1)}%
                </Descriptions.Item>
                <Descriptions.Item label="QD score">
                  {archiveData.metrics.qd_score.toFixed(2)}
                </Descriptions.Item>
                <Descriptions.Item label="Created">
                  {formatDate(archiveData.created_at)}
                </Descriptions.Item>
              </Descriptions>
              <Typography.Text type="secondary" style={{ marginTop: 12, display: 'block' }}>
                {archiveMode === 'frontier'
                  ? 'Showing the Pareto frontier bins for this run.'
                  : 'Showing the complete archive coverage for this run.'}
              </Typography.Text>
              <Table
                className="archive-card__table"
                columns={archiveColumns}
                dataSource={archiveData.bins}
                rowKey="bin_key"
                loading={archiveLoading}
                pagination={{ pageSize: 6 }}
                style={{ marginTop: 16 }}
                onRow={(record) => ({
                  onClick: () => setSelectedArchiveBin(record),
                })}
              />
              {selectedArchiveBin && (
                <Card type="inner" variant="borderless" style={{ marginTop: 16 }}>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="Bin">{selectedArchiveBin.bin_key}</Descriptions.Item>
                    <Descriptions.Item label="Build ID">
                      <Typography.Text copyable>{selectedArchiveBin.build_id}</Typography.Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="Score">
                      {selectedArchiveBin.score.toFixed(2)}
                    </Descriptions.Item>
                    <Descriptions.Item label="Descriptor">
                      <Space wrap size="small">
                        {Object.entries(selectedArchiveBin.descriptor).map(([axis, value]) => (
                          <Tag key={axis}>{`${axis}: ${value.toFixed(2)}`}</Tag>
                        ))}
                      </Space>
                    </Descriptions.Item>
                    <Descriptions.Item label="Metadata">
                      {Object.keys(selectedArchiveBin.metadata).length ? (
                        <List
                          size="small"
                          dataSource={Object.entries(selectedArchiveBin.metadata)}
                          renderItem={([key, value]) => (
                            <List.Item key={key}>
                              <Space align="baseline">
                                <Typography.Text strong>{key}</Typography.Text>
                                <Typography.Text code>{JSON.stringify(value)}</Typography.Text>
                              </Space>
                            </List.Item>
                          )}
                        />
                      ) : (
                        <Typography.Text type="secondary">No metadata</Typography.Text>
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="Tradeoff reasons">
                      {selectedArchiveBin.tradeoff_reasons?.length ? (
                        <Space wrap size="small">
                          {selectedArchiveBin.tradeoff_reasons.map((reason) => (
                            <Tag key={reason}>{reason}</Tag>
                          ))}
                        </Space>
                      ) : (
                        <Typography.Text type="secondary">Tradeoff metadata unavailable.</Typography.Text>
                      )}
                    </Descriptions.Item>
                    <Descriptions.Item label="Artifact links">
                      {normalizedArchiveArtifactLinks.length ? (
                        <Space direction="vertical">
                          {normalizedArchiveArtifactLinks.map((link) => (
                            <Typography.Link
                              key={link.label + link.url}
                              href={link.url}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {link.label}
                            </Typography.Link>
                          ))}
                        </Space>
                      ) : (
                        <Typography.Text type="secondary">No artifact links recorded.</Typography.Text>
                      )}
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              )}
            </>
          ) : (
            <Typography.Text type="secondary" style={{ marginTop: 16, display: 'block' }}>
              Provide a run ID and load an archive to inspect coverage.
            </Typography.Text>
          )}
        </Card>
      </Layout.Content>
      <Drawer
        open={drawerOpen}
        title={
          <Space>
            <Typography.Text strong>Build details</Typography.Text>
            {detail && <Tag color={statusColors[detail.build.status]}>{detail.build.status}</Tag>}
          </Space>
        }
        width={640}
        onClose={handleCloseDrawer}
        footer={
          detail ? (
            <Space>
              <Button
                icon={<EyeOutlined />}
                loading={xmlViewerLoading}
                onClick={() => handleViewXml(detail.build.build_id)}
              >
                View PoB XML
              </Button>
              <Button
                icon={<DownloadOutlined />}
                loading={downloadingXmlId === detail.build.build_id}
                onClick={() => handleDownloadXml(detail.build.build_id)}
              >
                Download XML
              </Button>
              <Button
                icon={<CopyOutlined />}
                loading={copyingCodeId === detail.build.build_id}
                onClick={() => handleCopyCode(detail.build.build_id)}
              >
                Copy PoB import code
              </Button>
            </Space>
          ) : null
        }
        destroyOnClose
      >
        {detailLoading && (
          <div className="drawer-spin">
            <Spin />
          </div>
        )}
        {detail && (
          <div className="drawer-content">
            <Descriptions column={1} size="small">
              <Descriptions.Item label="Build ID">{detail.build.build_id}</Descriptions.Item>
              <Descriptions.Item label="Profile">{detail.build.profile_id}</Descriptions.Item>
              <Descriptions.Item label="Ruleset">{detail.build.ruleset_id}</Descriptions.Item>
              <Descriptions.Item label="Class">{detail.build.class_}</Descriptions.Item>
              <Descriptions.Item label="Ascendancy">{detail.build.ascendancy}</Descriptions.Item>
              <Descriptions.Item label="Main skill">{detail.build.main_skill}</Descriptions.Item>
              <Descriptions.Item label="Created">
                {formatDate(detail.build.created_at)}
              </Descriptions.Item>
            </Descriptions>
            <Divider />
            <Typography.Title level={5}>Build composition</Typography.Title>
            {compositionDetails ? (
              <>
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="Source">{compositionSource ?? '—'}</Descriptions.Item>
                  <Descriptions.Item label="Class">
                    {(compositionIdentity?.class as string | undefined) ?? detail.build.class_}
                  </Descriptions.Item>
                  <Descriptions.Item label="Ascendancy">
                    {(compositionIdentity?.ascendancy as string | undefined) ?? detail.build.ascendancy}
                  </Descriptions.Item>
                  <Descriptions.Item label="Main skill">
                    {(compositionIdentity?.main_skill as string | undefined) ?? detail.build.main_skill}
                  </Descriptions.Item>
                  <Descriptions.Item label="Items captured">{compositionItems.length}</Descriptions.Item>
                  <Descriptions.Item label="Gem groups captured">
                    {compositionGemGroups.length}
                  </Descriptions.Item>
                  <Descriptions.Item label="Passive nodes captured">
                    {compositionPassiveNodes.length}
                  </Descriptions.Item>
                </Descriptions>

                <Typography.Text strong>Items</Typography.Text>
                {compositionItems.length ? (
                  <List
                    size="small"
                    dataSource={compositionItems}
                    renderItem={(item) => {
                      const slot =
                        (typeof item.slot === 'string' && item.slot) ||
                        (typeof item.slot_id === 'string' && item.slot_id) ||
                        'slot';
                      const name =
                        (typeof item.name === 'string' && item.name) ||
                        (typeof item.base_type === 'string' && item.base_type) ||
                        'item';
                      return (
                        <List.Item>
                          <Space direction="vertical" size={2} style={{ width: '100%' }}>
                            <Typography.Text strong>{slot}</Typography.Text>
                            <Typography.Text>{name}</Typography.Text>
                          </Space>
                        </List.Item>
                      );
                    }}
                  />
                ) : (
                  <Typography.Text type="secondary">No items captured.</Typography.Text>
                )}

                <Typography.Text strong>Gem groups</Typography.Text>
                {compositionGemGroups.length ? (
                  <List
                    size="small"
                    dataSource={compositionGemGroups}
                    renderItem={(group) => {
                      const groupName =
                        (typeof group.name === 'string' && group.name) ||
                        (typeof group.id === 'string' && group.id) ||
                        'group';
                      const gems = Array.isArray(group.gems)
                        ? group.gems
                            .map((gem) => {
                              if (typeof gem === 'string') {
                                return gem;
                              }
                              if (gem && typeof gem === 'object' && typeof gem.name === 'string') {
                                return gem.name;
                              }
                              return null;
                            })
                            .filter((gem): gem is string => gem !== null)
                        : [];
                      return (
                        <List.Item>
                          <Space direction="vertical" size={2} style={{ width: '100%' }}>
                            <Typography.Text strong>{groupName}</Typography.Text>
                            <Typography.Text>{gems.join(', ') || 'No gems listed'}</Typography.Text>
                          </Space>
                        </List.Item>
                      );
                    }}
                  />
                ) : (
                  <Typography.Text type="secondary">No gem groups captured.</Typography.Text>
                )}

                <Typography.Text strong>Passive nodes</Typography.Text>
                {compositionPassiveNodes.length ? (
                  <Typography.Paragraph className="scenario-json" code>
                    {JSON.stringify(compositionPassiveNodes, null, 2)}
                  </Typography.Paragraph>
                ) : (
                  <Typography.Text type="secondary">No passive nodes captured.</Typography.Text>
                )}
              </>
            ) : (
              <Typography.Text type="secondary">
                Build composition details are not yet available for this build.
              </Typography.Text>
            )}
            <Divider />
            <Typography.Title level={5}>Scenario metrics</Typography.Title>
            <Table
              columns={scenarioColumns}
              dataSource={detail.scenario_metrics}
              pagination={false}
              rowKey={(record) =>
                `${record.scenario_id}-${record.evaluated_at}-${record.gate_pass}-${record.full_dps}-${record.max_hit}`
              }
              size="small"
            />
            <Divider />
            <Typography.Title level={5}>Predicted vs verified</Typography.Title>
            {detailPrediction ? (
              <div className="prediction-detail">
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="State">
                    <Tag color={detailPredictionColor}>{detailPredictionState}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="Model">
                    {detailPrediction.model_id ?? '—'}
                    {detailPrediction.model_path && (
                      <Typography.Text type="secondary">
                        {detailPrediction.model_path}
                      </Typography.Text>
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="Full DPS">
                    <Space direction="vertical" size={0}>
                      <Typography.Text>
                        Predicted {formatPredictionValue(detailPrediction.predicted_full_dps)}
                      </Typography.Text>
                      <Typography.Text>
                        Verified {formatPredictionValue(detailPrediction.verified_full_dps)}
                      </Typography.Text>
                      {detailPrediction.error_full_dps != null && (
                        <Typography.Text type="secondary">
                          Error {detailPrediction.error_full_dps.toFixed(2)}
                        </Typography.Text>
                      )}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="Max Hit">
                    <Space direction="vertical" size={0}>
                      <Typography.Text>
                        Predicted {formatPredictionValue(detailPrediction.predicted_max_hit)}
                      </Typography.Text>
                      <Typography.Text>
                        Verified {formatPredictionValue(detailPrediction.verified_max_hit)}
                      </Typography.Text>
                      {detailPrediction.error_max_hit != null && (
                        <Typography.Text type="secondary">
                          Error {detailPrediction.error_max_hit.toFixed(2)}
                        </Typography.Text>
                      )}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="Pass probability">
                    {formatPredictionProbability(detailPrediction.pass_probability)} pass chance
                  </Descriptions.Item>
                  <Descriptions.Item label="Timestamp">
                    {detailPrediction.timestamp ?? '—'}
                  </Descriptions.Item>
                </Descriptions>
              </div>
            ) : (
              <Typography.Text type="secondary">
                Prediction metadata is not available for this build.
              </Typography.Text>
            )}
            <Divider />
            <Typography.Title level={5}>Cost breakdown</Typography.Title>
            {costDetails ? (
              <>
                <Descriptions column={2} size="small" bordered>
                  <Descriptions.Item label="Snapshot">
                    {costDetails.price_snapshot_id ?? '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="Total cost">
                    {costDetails.total_cost_chaos != null
                      ? costDetails.total_cost_chaos.toFixed(2) + 'c'
                      : '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="Unknown count">
                    {costDetails.unknown_cost_count ?? '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="Entries">
                    {costDetails.slot_costs.length + costDetails.gem_costs.length}
                  </Descriptions.Item>
                </Descriptions>
                <Typography.Text strong>Slot costs</Typography.Text>
                {costDetails.slot_costs.length ? (
                  <List
                    size="small"
                    dataSource={costDetails.slot_costs}
                    renderItem={(item) => (
                      <List.Item>
                        <Space direction="vertical" size={2} style={{ width: '100%' }}>
                          <Typography.Text strong>{item.slot ?? 'slot'}</Typography.Text>
                          <Typography.Text>{item.name}</Typography.Text>
                          <Space size="small">
                            <Typography.Text>
                              {item.cost_chaos != null
                                ? item.cost_chaos.toFixed(2) + 'c'
                                : 'unknown'}
                            </Typography.Text>
                            <Tag color={item.matched ? 'green' : 'volcano'}>
                              {item.matched ? 'priced' : 'unpriced'}
                            </Tag>
                          </Space>
                        </Space>
                      </List.Item>
                    )}
                  />
                ) : (
                  <Typography.Text type="secondary">
                    No slot costs recorded for this build.
                  </Typography.Text>
                )}
                <Typography.Title level={5} style={{ marginTop: 16 }}>
                  Gem costs
                </Typography.Title>
                {costDetails.gem_costs.length ? (
                  <List
                    size="small"
                    dataSource={costDetails.gem_costs}
                    renderItem={(item) => (
                      <List.Item>
                        <Space direction="vertical" size={2} style={{ width: '100%' }}>
                          <Typography.Text strong>{item.name}</Typography.Text>
                          <Typography.Text type="secondary">
                            Level {item.level ?? '—'} • Quality {item.quality ?? '—'}
                          </Typography.Text>
                          <Space size="small">
                            <Typography.Text>
                              {item.price != null ? item.price.toFixed(2) + 'c' : 'unknown'}
                            </Typography.Text>
                            <Tag color={item.matched ? 'green' : 'volcano'}>
                              {item.matched ? 'priced' : 'unpriced'}
                            </Tag>
                          </Space>
                        </Space>
                      </List.Item>
                    )}
                  />
                ) : (
                  <Typography.Text type="secondary">
                    No gem costs recorded for this build.
                  </Typography.Text>
                )}
              </>
            ) : (
              <Typography.Text type="secondary">
                Cost breakdown appears once pricing data is available.
              </Typography.Text>
            )}
            <Divider />
            <Typography.Title level={5}>Scenario assumptions</Typography.Title>
            {detail.scenarios_used.length ? (
              <List
                size="small"
                dataSource={detail.scenarios_used}
                renderItem={(scenario) => (
                  <List.Item>
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Typography.Text strong>
                        {scenario.scenario_id} ({scenario.version})
                      </Typography.Text>
                      <Typography.Text type="secondary">
                        Profile: {scenario.profile_id}
                      </Typography.Text>
                      <Typography.Paragraph className="scenario-json" code>
                        {JSON.stringify(scenario.pob_config, null, 2)}
                      </Typography.Paragraph>
                      <Typography.Paragraph className="scenario-json" code>
                        {JSON.stringify(scenario.gate_thresholds, null, 2)}
                      </Typography.Paragraph>
                    </Space>
                  </List.Item>
                )}
              />
            ) : (
              <Typography.Text type="secondary">
                No scenario assumptions were persisted for this build.
              </Typography.Text>
            )}
            <Divider />
            <Typography.Title level={5}>Scenario diff</Typography.Title>
            {scenarioDiff ? (
              <>
                <Space wrap>
                  <Select
                    value={leftScenarioId}
                    style={{ minWidth: 200 }}
                    options={scenarioIds.map((value) => ({ label: value, value }))}
                    onChange={(value) => setLeftScenarioId(value)}
                  />
                  <Select
                    value={rightScenarioId}
                    style={{ minWidth: 200 }}
                    options={scenarioIds.map((value) => ({ label: value, value }))}
                    onChange={(value) => setRightScenarioId(value)}
                  />
                </Space>
                <Typography.Text type="secondary">
                  Showing top {Math.min(5, scenarioDiff.rows.length)} metric deltas by absolute
                  change.
                </Typography.Text>
                <Table
                  size="small"
                  pagination={false}
                  rowKey="key"
                  dataSource={scenarioDiff.rows.slice(0, 5)}
                  columns={[
                    {
                      title: 'Metric',
                      dataIndex: 'metric',
                      key: 'metric',
                    },
                    {
                      title: scenarioDiff.leftScenario,
                      dataIndex: 'leftValue',
                      key: 'leftValue',
                      render: (value: number) => value.toFixed(2),
                    },
                    {
                      title: scenarioDiff.rightScenario,
                      dataIndex: 'rightValue',
                      key: 'rightValue',
                      render: (value: number) => value.toFixed(2),
                    },
                    {
                      title: 'Delta',
                      dataIndex: 'delta',
                      key: 'delta',
                      render: (value: number) => value.toFixed(2),
                    },
                  ]}
                />
              </>
            ) : (
              <Typography.Text type="secondary">
                Scenario diff appears when at least two distinct scenarios are available for this
                build.
              </Typography.Text>
            )}
            <Divider />
            <Typography.Title level={5}>Constraint verdict</Typography.Title>
            {hasConstraintMetadata ? (
              <>
                <Descriptions column={1} size="small" bordered>
                  <Descriptions.Item label="Status">
                    {detailConstraintStatus ? (
                      <Tag color={constraintStatusColors[detailConstraintStatus]}>
                        {detailConstraintStatus}
                      </Tag>
                    ) : (
                      <Typography.Text type="secondary">Not evaluated</Typography.Text>
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="Reason code">
                    {detailConstraintReason ?? '—'}
                  </Descriptions.Item>
                  <Descriptions.Item label="Checked at">
                    {formatDate(detailConstraintCheckedAt ?? undefined)}
                  </Descriptions.Item>
                </Descriptions>
                <Typography.Text strong>Violated constraints</Typography.Text>
                {detailViolatedConstraints.length ? (
                  <List
                    size="small"
                    dataSource={detailViolatedConstraints}
                    renderItem={(item) => <List.Item>{item}</List.Item>}
                  />
                ) : (
                  <Typography.Text type="secondary">
                    No constraint violations recorded.
                  </Typography.Text>
                )}
              </>
            ) : (
              <Typography.Text type="secondary">
                Constraint metadata is not available for this build.
              </Typography.Text>
            )}
            <Divider />
            <Space direction="vertical" size="small">
              <div>
                <Typography.Text strong>Gate fail reasons</Typography.Text>
                {gateFailures.length ? (
                  <List
                    size="small"
                    dataSource={gateFailures}
                    renderItem={(reason) => <List.Item>{reason}</List.Item>}
                  />
                ) : (
                  <Typography.Text type="secondary">No gate failures reported.</Typography.Text>
                )}
              </div>
              <div>
                <Typography.Text strong>Warnings</Typography.Text>
                {warnings.length ? (
                  <List
                    size="small"
                    dataSource={warnings}
                    renderItem={(item) => <List.Item>{item}</List.Item>}
                  />
                ) : (
                  <Typography.Text type="secondary">No warnings surfaced.</Typography.Text>
                )}
              </div>
            </Space>
          </div>
        )}
      </Drawer>
      <Modal
        title="PoB XML Viewer"
        open={xmlViewerOpen}
        onCancel={handleCloseXmlViewer}
        width={800}
        footer={[
          <Button key="download" icon={<DownloadOutlined />} onClick={() => detail && handleDownloadXml(detail.build.build_id)}>
            Download
          </Button>,
          <Button key="close" type="primary" onClick={handleCloseXmlViewer}>
            Close
          </Button>,
        ]}
        destroyOnClose
      >
        {xmlViewerLoading ? (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin />
          </div>
        ) : xmlViewerContent ? (
          <pre style={{ 
            maxHeight: 500, 
            overflow: 'auto', 
            backgroundColor: '#f5f5f5', 
            padding: 12, 
            fontSize: 11,
            fontFamily: 'monospace'
          }}>
            {xmlViewerContent}
          </pre>
        ) : (
          <Typography.Text type="secondary">No XML content available</Typography.Text>
        )}
      </Modal>
    </Layout>
  );
};

export default App;
