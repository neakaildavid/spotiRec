/** Shimmering placeholders shown while data loads (instead of spinners). */

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function SongCardSkeleton() {
  return (
    <div className="rounded-xl bg-surface p-3" aria-hidden="true">
      <Skeleton className="aspect-square w-full rounded-lg" />
      <Skeleton className="mt-3 h-3.5 w-3/4" />
      <Skeleton className="mt-2 h-3 w-1/2" />
    </div>
  );
}

export function TrackRowSkeleton() {
  return (
    <div className="grid grid-cols-[2rem_minmax(0,1fr)_3.5rem] items-center gap-4 px-3 py-2 md:grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)_4rem]" aria-hidden="true">
      <Skeleton className="h-3 w-4" />
      <div className="flex items-center gap-3">
        <Skeleton className="size-10 shrink-0" />
        <div className="w-full">
          <Skeleton className="h-3.5 w-2/3" />
          <Skeleton className="mt-2 h-3 w-1/3 md:hidden" />
        </div>
      </div>
      <Skeleton className="hidden h-3 w-1/2 md:block" />
      <Skeleton className="ml-auto h-3 w-8" />
    </div>
  );
}
