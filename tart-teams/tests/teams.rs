#![allow(clippy::unwrap_used, reason = "test assertions may unwrap")]

use std::thread;
use std::time::Duration;

use tart_teams::{Child, Kind, Message, Permissions, Policy, SendError, Team};

#[test]
fn reports_flow_up_and_ids_are_minted() {
    let (team, lead) = Team::new(Permissions::open());
    let first = team.spawn_child();
    let second = team.spawn_child();
    assert_ne!(first.id(), second.id());
    assert_eq!(first.peers().count(), 1);
    first.report("done").unwrap();
    let env = lead.inbox().recv().unwrap();
    assert_eq!(env.from, first.id().id());
    assert_eq!(env.message, Message::Data("done".into()));
}

#[test]
fn steering_flows_down() {
    let (team, lead) = Team::new(Permissions::open());
    let child = team.spawn_child();
    lead.steer(child.id(), "focus").unwrap();
    let env = child.inbox().recv().unwrap();
    assert_eq!(env.from, 0);
    assert_eq!(env.message, Message::Steer("focus".into()));
}

#[test]
fn cancellation_can_be_disabled() {
    let (team, lead) = Team::new(Permissions::open().without_cancel_between(0, 1));
    let child = team.spawn_child();
    assert_eq!(
        lead.cancel(child.id(), "stop"),
        Err(SendError::Forbidden { from: 0, to: 1, kind: Kind::Cancel })
    );
    assert!(child.inbox().is_empty(), "never delivered");
    lead.steer(child.id(), "steering still flows").unwrap();
    let gated = Team::new(Permissions::open().without_cancel_from(0));
    let other = gated.0.spawn_child();
    assert!(gated.1.cancel(other.id(), "no").is_err());
    assert!(other.inbox().is_empty());
}

#[test]
fn reports_pass_the_policy_gate_too() {
    struct Mute;
    impl Policy for Mute {
        fn permits(&self, _: u64, _: u64, kind: Kind) -> bool {
            !matches!(kind, Kind::Data)
        }
    }
    let (team, lead) = Team::new(Mute);
    let child = team.spawn_child();
    assert!(child.report("silenced").is_err());
    assert!(lead.inbox().is_empty());
}

#[test]
fn dropping_a_child_retires_its_id() {
    let (team, lead) = Team::new(Permissions::open());
    let child = team.spawn_child();
    let id = child.id();
    drop(child);
    assert_eq!(lead.steer(id, "gone"), Err(SendError::Gone(1)));
}

#[test]
fn broadcast_respects_policy() {
    let (team, lead) = Team::new(Permissions::open());
    let kids: Vec<Child> = (0..3).map(|_| team.spawn_child()).collect();
    let steered = lead.broadcast(&Message::Steer("pivot".into()));
    assert_eq!(steered.len(), 3);
    for (_, result) in steered {
        result.unwrap();
    }
    assert!(kids.iter().all(|kid| kid.inbox().len() == 1));

    let (gated, gated_lead) = Team::new(Permissions::open().without_cancel_from(0));
    let gated_kids: Vec<Child> = (0..2).map(|_| gated.spawn_child()).collect();
    let cancelled = gated_lead.broadcast(&Message::Cancel("stop".into()));
    assert_eq!(cancelled.len(), 2);
    assert!(cancelled.iter().all(|(_, result)| result.is_err()));
    assert!(
        gated_kids.iter().all(|kid| kid.inbox().is_empty()),
        "a gated broadcast delivers nothing"
    );
}

#[test]
fn peers_exclude_self_and_the_retired() {
    let (team, _lead) = Team::new(Permissions::open());
    let one = team.spawn_child();
    let two = team.spawn_child();
    let three = team.spawn_child();
    drop(three);
    let peers: Vec<_> = one.peers().collect();
    assert_eq!(peers, vec![two.id()]);
}

#[test]
fn lateral_sends_flow_and_main_is_not_addressable() {
    let (team, lead) = Team::new(Permissions::open());
    let one = team.spawn_child();
    let two = team.spawn_child();
    one.tell(two.id(), "status").unwrap();
    one.steer_peer(two.id(), "focus").unwrap();
    let told = two.inbox().recv().unwrap();
    assert_eq!((told.from, told.message), (1, Message::Data("status".into())));
    let steered = two.inbox().recv().unwrap();
    assert_eq!(
        (steered.from, steered.message),
        (1, Message::Steer("focus".into()))
    );
    assert!(lead.inbox().is_empty(), "MAIN is not addressable sideways");
}

#[test]
fn self_addressed_peer_sends_are_refused() {
    let (team, _lead) = Team::new(Permissions::open());
    let one = team.spawn_child();
    assert!(matches!(
        one.tell(one.id(), "note"),
        Err(SendError::Forbidden { from: 1, to: 1, .. })
    ));
    assert!(one.inbox().is_empty());
}

#[test]
fn lateral_send_to_a_retired_child_is_gone() {
    let (team, _lead) = Team::new(Permissions::open());
    let one = team.spawn_child();
    let ghost = team.spawn_child();
    let ghost_id = ghost.id();
    drop(ghost);
    assert_eq!(one.tell(ghost_id, "ping"), Err(SendError::Gone(2)));
}

#[test]
fn cancel_stops_a_worker_thread() {
    let (team, lead) = Team::new(Permissions::open());
    let child = team.spawn_child();
    let id = child.id();
    thread::scope(|s| {
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        s.spawn(move || {
            while let Ok(env) = child.inbox().recv_timeout(Duration::from_secs(5)) {
                if matches!(env.message, Message::Cancel(_)) {
                    break;
                }
            }
            done_tx.send(()).unwrap();
        });
        lead.cancel(id, "wrap up").unwrap();
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("worker stopped after cancel");
    });
}
